import os
import argparse
import pickle as pk
import numpy as np
import time

# import tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from stgcn_att import STGCN
from utils import generate_dataset, load_metr_la_data, get_normalized_adj, load_2_metr_la_data


use_gpu = False
num_timesteps_input = 12
num_timesteps_output = 3

epochs = 100
batch_size = 32

parser = argparse.ArgumentParser(description='STGCN')
parser.add_argument('--enable-cuda', action='store_true',
                    help='Enable CUDA')
args = parser.parse_args()
args.device = None
args.enable_cuda = False
if args.enable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
    print("cuda")
else:
    args.device = torch.device('cpu')
    print("cpu")


def train_epoch(training_input, training_target, batch_size):
    """
    Trains one epoch with the given data.experiment_data
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    permutation = torch.randperm(training_input.shape[0])

    epoch_training_losses = []
    for i in range(0, training_input.shape[0], batch_size):
        net.train()
        optimizer.zero_grad()  # initialize the grad as 0

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        X_batch = X_batch.to(device=args.device)
        y_batch = y_batch.to(device=args.device)

        out = net(X_batch) # fed data here
        loss = loss_criterion(out, y_batch)
        loss.backward()  # calculate grad
        optimizer.step()  # update parameters
        epoch_training_losses.append(loss.detach().cpu().numpy())
    return sum(epoch_training_losses)/len(epoch_training_losses)


if __name__ == '__main__':
    torch.manual_seed(7)
    A, X, means, stds = load_metr_la_data() #ajmatrix, data
    #A, A_2, X, means, stds = load_2_metr_la_data()

    split_line1 = int(X.shape[2] * 0.6)
    split_line2 = int(X.shape[2] * 0.8)

    train_original_data = X[:, :, :split_line1]
    val_original_data = X[:, :, split_line1:split_line2]
    test_original_data = X[:, :, split_line2:]

    training_input, training_target = generate_dataset(train_original_data,
                                                       num_timesteps_input=num_timesteps_input,
                                                       num_timesteps_output=num_timesteps_output)
    val_input, val_target = generate_dataset(val_original_data,
                                             num_timesteps_input=num_timesteps_input,
                                             num_timesteps_output=num_timesteps_output)
    test_input, test_target = generate_dataset(test_original_data,
                                               num_timesteps_input=num_timesteps_input,
                                               num_timesteps_output=num_timesteps_output)

    A_wave = get_normalized_adj(A)  # check
    A_wave = torch.from_numpy(A_wave)
    A_wave = A_wave.to(device=args.device)
    
    #A_wave_2 = get_normalized_adj(A_2)
    #A_wave_2 = torch.from_numpy(A_wave_2)
    #A_wave_2 = A_wave_2.to(device=args.device)

    net = STGCN(A_wave.shape[0], # number of row : nodes(station)
                training_input.shape[3], # num of features
                num_timesteps_input,
                num_timesteps_output,
                A_wave)

    net.double()  # parameter calculated as double

    # optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.001)
    loss_criterion = nn.MSELoss()

    training_losses = []
    validation_losses = []
    validation_maes = []
    best_mae = 100
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.5, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 100, eta_min=1e-6, last_epoch=-1)
    for epoch in range(epochs):
        print("****************epoch {}******************".format(epoch+1))
        now = time.time()
        

        loss = train_epoch(training_input, training_target,
                           batch_size=batch_size)
        scheduler.step()  # update the learning rate
        training_losses.append(loss)
        # Run validation

        with torch.no_grad():  # don't calculate the grad when evaluation
            net.eval()
            val_input = val_input.to(device=args.device)
            val_target = val_target.to(device=args.device)

            out = net(val_input)
            val_loss = loss_criterion(out, val_target).to(device="cpu")
            validation_losses.append(np.asscalar(val_loss.detach().numpy()))

            out_unnormalized = out.detach().cpu().numpy()*stds[0]+means[0]  # 0 is the index of available bike
            target_unnormalized = val_target.detach().cpu().numpy()*stds[0]+means[0]
            # print("out_unnormalized: {}".format(out_unnormalized))
            # print("target_unnormalized: {}".format(target_unnormalized))
            mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))
            validation_maes.append(mae)

            out = None
            val_input = val_input.to(device="cpu")
            val_target = val_target.to(device="cpu")

        print("Training loss: {}".format(training_losses[-1]))
        print("Validation loss: {}".format(validation_losses[-1]))
        print("Validation MAE: {}".format(validation_maes[-1]))
        # plt.plot(training_losses, label="training loss")
        # plt.plot(validation_losses, label="validation loss")
        # plt.legend()
        # plt.show()
        print("Time consume: {}s".format(time.time()-now))

        checkpoint_path = "checkpoints/"
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        if validation_maes[-1] < best_mae:
            best_mae = validation_maes[-1]
            torch.save(net, "checkpoints/"+"/"+'best_model'+str(best_mae)+'.pkl')
        with open("checkpoints/losses.pk", "wb") as fd:
            pk.dump((training_losses, validation_losses, validation_maes), fd)
    plt.plot(training_losses, label="training loss")
    plt.plot(validation_losses, label="validation loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(validation_maes, label="validation_maes")
    plt.legend()
    plt.show()
