import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# d_prob=0.5

def DotProductAttention(decode_q, encode_k):
    att = decode_q.permute()*encode_k
    return att

#self.att_weight = torch.Tensor(np.ones([2,2,3,3]))
#self.att_weight=torch.nn.Parameter(self.att_weight)
# def BilinearAttention(att_weight, decode_q, encode_k):
#     return 

# def ScaledDotProductAttention(decode_q, encode_k):

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
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0,1))
        self.dropout = nn.Dropout2d(p=0.5)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0,1))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0,1))

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
        # print("****")
        # print(X.shape)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        # print(temp.shape)
        out = F.relu(temp + self.conv3(X))
        # print(out.shape)

        # temp = self.dropout(self.conv1(X)) + torch.sigmoid(self.dropout(self.conv2(X)))
        # out = F.relu(temp + self.dropout(self.conv3(X)))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out

# torch.Size([32, 8, 110, 12])
# torch.Size([32, 64, 110, 10])
# torch.Size([32, 64, 110, 10])
# ****
# torch.Size([32, 16, 110, 10])
# torch.Size([32, 64, 110, 8])
# torch.Size([32, 64, 110, 8])
# ****
# torch.Size([32, 64, 110, 8])
# torch.Size([32, 64, 110, 6])
# torch.Size([32, 64, 110, 6])
# ****
# torch.Size([32, 16, 110, 6])
# torch.Size([32, 64, 110, 4])
# torch.Size([32, 64, 110, 4])
# ****
# torch.Size([32, 64, 110, 4])
# torch.Size([32, 64, 110, 2])
# torch.Size([32, 64, 110, 2])



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
        self.avg_pool_3d_1 = nn.AdaptiveAvgPool3d(1)
        self.avg_pool_3d_2 = nn.AdaptiveAvgPool3d(1)
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
        t = self.temporal1(X)#32, 110, 12, 64
        p1 = self.avg_pool_3d_1(t.permute(2, 0, 1, 3)).reshape(12)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))#32, 110, 12, 16
        t3 = self.temporal2(t2)#32, 110, 12, 64
        p2 = self.avg_pool_3d_2(t3.permute(2, 0, 1, 3)).reshape(12)
        att = F.sigmoid(F.relu(p1+p2))
        t4 = torch.mul(t.permute(0,1,3,2),att).permute(0,1,3,2)
        t5 = torch.cat((t4, t3), dim = 3)
        return self.batch_norm(t5)
        # return t3


class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, A_hat):
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
        self.block2 = STGCNBlock(in_channels=64*2, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=64*2, out_channels=64)
        self.fully = nn.Linear((num_timesteps_input) * 64,
                               num_timesteps_output)
        # self.fully = nn.Linear((num_timesteps_input - 2 * 5) * 64,
        #                        num_timesteps_output)
        # self.A_hat = nn.Parameter(A_hat)
        self.A_hat = A_hat

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        4. Coupled layer-wise: SVD(Xa, n) = Xt*Xs, Aij = exp(-(d/(sigma*sigma)))
        original n = 20, sigma = 1(set by myself)
        """
        # X:
        # torch.Size([32, 110, 12, 8]) batch size, n_node, time_s, feature
        # A_hat:
        # torch.Size([110, 110])
        # out1:
        # torch.Size([32, 110, 8, 64])
        # out2:
        # torch.Size([32, 110, 4, 64])
        # out3:
        # torch.Size([32, 110, 2, 64])
        # out4:
        # torch.Size([32, 110, 3])

        # A_hat is also an initial attention matrix or the attention values
        X_a = X.reshape(-1, X.shape[1])
        u, s, v = torch.svd(X_a)
        X_s = u[0:20,:]
        for i in range(X.shape[1]):
            for j in range(X.shape[1]):
                
                self.A_hat[i,j] = torch.exp(F.pairwise_distance(X_s[:,i].view(1,-1), X_s[:,j].view(1,-1), p=2))
        out1 = self.block1(X, self.A_hat) # Time-attention
        out2 = self.block2(out1, self.A_hat)# Space-attention
        out3 = self.last_temporal(out2)
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        return out4

# self.A_hat
# class STGCN(nn.Module):
#     """
#     Spatio-temporal graph convolutional network as described in
#     https://arxiv.org/abs/1709.04875v3 by Yu et al.
#     Input should have shape (batch_size, num_nodes, num_input_time_steps,
#     num_features).
#     """

#     def __init__(self, num_nodes, num_features, num_timesteps_input,
#                  num_timesteps_output, A_hat):
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
#         self.A_hat = nn.Parameter(A_hat)

#     def forward(self, X):
#         """
#         :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
#         num_features=in_channels).
#         :param A_hat: Normalized adjacency matrix.
#         """
#         out1 = self.block1(X, self.A_hat)
#         out2 = self.block2(out1, self.A_hat)
#         out3 = self.last_temporal(out2)
#         out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
#         return out4

# Original
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


