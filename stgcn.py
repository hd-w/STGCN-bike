import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# d_prob=0.5

class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        # self.dropout = nn.Dropout2d(p=0.5)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        # X = X.double()
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))

        # temp = self.dropout(self.conv1(X)) + torch.sigmoid(self.dropout(self.conv2(X)))
        # out = F.relu(temp + self.dropout(self.conv3(X)))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)
        # return t3

class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, A_hat, A_hat_2):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN, self).__init__()
        self.block1 = STGCNBlock(in_channels=num_features, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.block2 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        self.fully = nn.Linear((num_timesteps_input - 2 * 5) * 64 * 2,
                               num_timesteps_output)
        self.A_hat = nn.Parameter(A_hat)
        self.A_hat_2 = nn.Parameter(A_hat_2)

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        out1 = self.block1(X, self.A_hat)
        out2 = self.block2(out1, self.A_hat)
        out3 = self.last_temporal(out2)
        # print(out3.shape)
        
        out5 = self.block1(X, self.A_hat_2)
        out6 = self.block2(out5, self.A_hat_2)
        out7 = self.last_temporal(out6)
        
        out8 = torch.cat((out3, out7),2)
        x_3 = out3.reshape((out3.shape[0], out3.shape[1], -1))
        x_8 = out8.reshape((out8.shape[0], out8.shape[1], -1))
        # print(out8.shape)
        # print(x_3.shape)
        # print(x_8.shape)
        out4 = self.fully(out8.reshape((out8.shape[0], out8.shape[1], -1)))
        return out4

# class STGCN(nn.Module):
#     """
#     Spatio-temporal graph convolutional network as described in
#     https://arxiv.org/abs/1709.04875v3 by Yu et al.
#     Input should have shape (batch_size, num_nodes, num_input_time_steps,
#     num_features).
#     """

#     def __init__(self, num_nodes, num_features, num_timesteps_input,
#                  num_timesteps_output):
#         """
#         :param num_nodes: Number of nodes in the graph.
#         :param num_features: Number of features at each node in each time step.
#         :param num_timesteps_input: Number of past time steps fed into the
#         network.
#         :param num_timesteps_output: Desired number of future time steps
#         output by the network.
#         """
#         super(STGCN, self).__init__()
#         self.block1 = STGCNBlock(in_channels=num_features, out_channels=64,
#                                  spatial_channels=16, num_nodes=num_nodes)
#         self.block2 = STGCNBlock(in_channels=64, out_channels=64,
#                                  spatial_channels=16, num_nodes=num_nodes)
#         self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
#         self.fully = nn.Linear((num_timesteps_input - 2 * 5) * 64,
#                                num_timesteps_output)

#     def forward(self, A_hat, X):
#         """
#         :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
#         num_features=in_channels).
#         :param A_hat: Normalized adjacency matrix.
#         """
#         out1 = self.block1(X, A_hat)
#         out2 = self.block2(out1, A_hat)
#         out3 = self.last_temporal(out2)
#         out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
#         return out4


