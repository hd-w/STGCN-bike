import os
import argparse
import pickle as pk
import numpy as np
import time

import tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import patheffects

import torch
import torch.nn as nn

from stgcn_a3 import STGCN
from utils import generate_dataset, load_metr_la_data, get_normalized_adj, load_2_metr_la_data
import csv


use_gpu = True
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

def plotting(outputs, targets):
    plot_id = 2#Blessington Street

    fig = plt.figure(figsize=(25,13))

    font1 = {'family' : 'Times New Roman',
            'weight' : 'bold',
            'size' : 16,
            }
    font2 = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size' : 16,
            }
    font3 = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size' : 19,
            }

    plt.subplot(2,1,1)
    plt.plot(outputs[:, plot_id, 0], label="Prediction")
    plt.plot(targets[:, plot_id, 0], label="Ground Truth")
    plt.ylabel('Average Number of Available Bicycles', font1)
    plt.xlabel('Time(15mins/time stamp)', font1)
    title_text_obj=plt.title('First time stamp prediction and ground truth',fontsize=23,verticalalignment='bottom',fontweight='bold')
    plt.xticks(fontproperties = 'Times New Roman', size = 14, weight = 'bold')
    plt.yticks(fontproperties = 'Times New Roman', size = 14, weight = 'bold')

    
    plt.legend(prop=font1)
    # plt.show()

    plt.subplot(2,1,2)
    # plt.plot(outputs[:, plot_id, 1], label="Prediction")
    # plt.plot(targets[:, plot_id, 1], label="Ground Truth")
    # plt.ylabel('Average Number of Available Bicycles')
    # plt.xlabel('Time Stamp')
    # plt.legend()
    # plt.show()

    plt.plot(outputs[:, plot_id, 2], label="Prediction")
    plt.plot(targets[:, plot_id, 2], label="Ground Truth")
    plt.ylabel('Average Number of Available Bicycles', font1)
    plt.xlabel('Time(15mins/time stamp)', font1)
    title_text_obj=plt.title('Third time stamp prediction and ground truth',fontsize=23,verticalalignment='bottom',fontweight='bold')
    plt.xticks(fontproperties = 'Times New Roman', size = 14, weight = 'bold')
    plt.yticks(fontproperties = 'Times New Roman', size = 14, weight = 'bold')

    plt.legend(prop=font1)
    plt.tight_layout()
    plt.show()


def main():
    A, X, means, stds = load_metr_la_data()
    #A, A_2, X, means, stds = load_2_metr_la_data()

    print('X shape is ',X.shape)
    split_line1 = int(X.shape[2] * 0.6)
    split_line2 = int(X.shape[2] * 0.8)

    test_original_data = X[:, :, split_line2:]

    test_inputs, test_targets = generate_dataset(test_original_data,
                                               num_timesteps_input=num_timesteps_input,
                                               num_timesteps_output=num_timesteps_output)

    print('test_inputs shape is ',test_inputs.shape)
    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave)
    A_wave = A_wave.to(device=args.device)

    net = torch.load("./checkpoints/best_model1.00.pkl", map_location='cpu')

    outputs = np.zeros((1750,110,3))
    targets = np.zeros((1750,110,3))
    #np.abs(target(:,0,0)-output(:,0,0))==mae of node 1 first prediciton
    
    with torch.no_grad():    
        net.eval()
        for i in range(175):
            print("****************epoch {}******************".format(i+1)) 
            test_input = test_inputs[i*10:(i+1)*10,:,:,:].to(device=args.device)
            test_target = test_targets[i*10:(i+1)*10,:,:].to(device=args.device)

            # print(test_input.shape)
            out = net(test_input)

            out_unnormalized = out.detach().cpu().numpy()*stds[0]+means[0]
            target_unnormalized = test_target.detach().cpu().numpy()*stds[0]+means[0]
            # print("out_unnormalized: {}".format(out_unnormalized))
            # print("target_unnormalized: {}".format(target_unnormalized))
            mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))
            print(mae)

            outputs[i*10:(i+1)*10,:,:] = out_unnormalized
            targets[i*10:(i+1)*10,:,:] = target_unnormalized
    
    plotting(outputs, targets)
    print('targets shape is ',targets.shape) 
    
    # f = open('stationmae.csv','w',encoding='utf-8')
    # csv_writer = csv.writer(f)
    # csv_writer.writerow(["Station ID","MAE"])
    # for i in range(110):
    #     mae_station = np.mean(np.abs(targets[:,i,:] - outputs[:,i,:]))
    #     csv_writer.writerow([str(i+1),str(mae_station)])
    # f.close()


main()


