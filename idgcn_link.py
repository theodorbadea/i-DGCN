import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def process(mul_L_real, weight, X_real):
    data = torch.spmm(mul_L_real, X_real)
    real = torch.matmul(data, weight) 
    return torch.stack([real])

class ChebConv(nn.Module):
    """
    The MagNet convolution operation.

    :param in_c: int, number of input channels.
    :param out_c: int, number of output channels.
    :param K: int, the order of Chebyshev Polynomial.
    :param L_norm_real, L_norm_imag: normalized laplacian of real and imag
    """
    def __init__(self, in_c, out_c, K,  L_norm_real, bias=True):
        super(ChebConv, self).__init__()

        self.K = K
        self.L_norm_real = L_norm_real

        self.weight = nn.Parameter(torch.Tensor(K, in_c, out_c))

        stdv = 1. / math.sqrt(self.weight.size(-1))
        self.weight.data.uniform_(-stdv, stdv)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_c))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, data):
        x = data
        poly = []
        if self.K < 0:
            raise ValueError('ERROR: K must be non-negative!')
        elif self.K == 0:
            # x_0 = x
            poly.append(x)
        elif self.K == 1:
            # x_0 = x
            poly.append(x)
            if self.L_norm_real.is_sparse:
                # x_1 = L * x
                poly.append(torch.sparse.mm(self.L_norm_real, x))
            else:
                if x.is_sparse:
                    x = x.to_dense
                # x_1 = L * x
                poly.append(torch.mm(self.L_norm_real, x))
        else:
            # x_0 = x
            poly.append(x)
            # x_1 = L * x
            poly.append(torch.mm(self.L_norm_real, x))
            # x_k = 2 * L * x_{k-1} - x_{k-2}
            for k in range(2, self.K):
                poly.append(torch.mm(2 * self.L_norm_real, poly[k - 1]) - poly[k - 2])
        
        feature = torch.stack(poly, dim=0)
        if feature.is_sparse:
            feature = feature.to_dense()
        graph_conv = torch.einsum('bij,bjk->ik', feature, self.weight)

        if self.bias != None:
            graph_conv = torch.add(input=graph_conv, other=self.bias, alpha=1)

        return graph_conv

class ChebNet_Edge(nn.Module):
    def __init__(self, in_c, L_norm_real, num_filter=2, K=2, label_dim=2, activation=True, layer=2, dropout=0):
        """
        :param in_c: int, number of input channels.
        :param hid_c: int, number of hidden channels.
        :param K: for cheb series
        :param L_norm_real, L_norm_imag: normalized laplacian
        """
        super(ChebNet_Edge, self).__init__()
        
        chebs = [ChebConv(in_c=in_c, out_c=num_filter, K=K, L_norm_real=L_norm_real)]
        if activation and (layer != 1):
            chebs.append(torch.nn.ReLU())

        for i in range(1, layer):
            chebs.append(ChebConv(in_c=num_filter, out_c=num_filter, K=K, L_norm_real=L_norm_real))
            if activation:
                chebs.append(torch.nn.ReLU())
        self.Chebs = torch.nn.Sequential(*chebs)
        
        last_dim = 2
        self.linear = nn.Linear(num_filter*last_dim, label_dim)     
        self.dropout = dropout

    def forward(self, real, index):
        real = self.Chebs((real))
        x = torch.cat((real[index[:,0]], real[index[:,1]]), dim = -1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x