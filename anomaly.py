import torch.nn as nn
import torch.nn.functional as F
import math
import torch

class ChebyshevConv(torch.nn.Module):
    def __init__(self, K, size_in, size_out, bias=None):
        super(ChebyshevConv, self).__init__()

        self.K = K
        self.weight = torch.nn.Parameter(torch.FloatTensor(K, size_in, size_out))

        if bias != None:
            self.bias = torch.nn.Parameter(torch.FloatTensor(size_out))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias != None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, L):
        # x_0 = x,
        # x_1 = L * x,
        # x_k = 2 * L * x_{k-1} - x_{k-2},
        # where L = 2 * L / eigv_max - I.

        poly = []
        if self.K < 0:
            raise ValueError('ERROR: K must be non-negative!')
        elif self.K == 0:
            # x_0 = x
            poly.append(x)
        elif self.K == 1:
            # x_0 = x
            poly.append(x)
            if L.is_sparse:
                # x_1 = L * x
                poly.append(torch.sparse.mm(L, x))
            else:
                if x.is_sparse:
                    x = x.to_dense
                # x_1 = L * x
                poly.append(torch.mm(L, x))
        else:
            # x_0 = x
            poly.append(x)
            # x_1 = L * x
            poly.append(torch.mm(L, x))
            # x_k = 2 * L * x_{k-1} - x_{k-2}
            for k in range(2, self.K):
                poly.append(torch.mm(2 * L, poly[k - 1]) - poly[k - 2])
        
        feature = torch.stack(poly, dim=0)
        if feature.is_sparse:
            feature = feature.to_dense()
        graph_conv = torch.einsum('bij,bjk->ik', feature, self.weight)

        if self.bias != None:
            graph_conv = torch.add(input=graph_conv, other=self.bias, alpha=1)

        return graph_conv

class ChebGCN(nn.Module):
    def __init__(self, size_in, size_out, hidden_dim, nb_layers, K, enable_bias=False, droprate=None):
        super(ChebGCN, self).__init__()

        self.layers = nn.ModuleList()

        # additional linear layer to handle dimension discrepancies (if any)
        if size_in != hidden_dim:
            self.layers.append(nn.Linear(in_features=size_in, out_features=hidden_dim, bias=enable_bias))
        
        # encoder
        for _ in range(nb_layers):
            new_dim = int(hidden_dim / 2)
            self.layers.append(ChebyshevConv(K, hidden_dim, new_dim, enable_bias))
            hidden_dim = new_dim

        # decoder
        for _ in range(nb_layers):
            new_dim = int(hidden_dim * 2)
            self.layers.append(ChebyshevConv(K, hidden_dim, new_dim, enable_bias))
            hidden_dim = new_dim
        
        # additional linear layer to handle dimension discrepancies (if any) 
        if size_in != hidden_dim:
            self.layers.append(nn.Linear(in_features=hidden_dim, out_features=size_out, bias=enable_bias))

        if droprate != None:
            self.dropout = nn.Dropout(p=droprate)
        else:
            self.dropout = None
        
        self.activation = nn.ReLU()

    def forward(self, x, L):
        for i in range(len(self.layers)):
            if type(self.layers[i]) == nn.Linear:
                x = self.layers[i](x)
            else:
                x = self.layers[i](x, L)
                x = self.activation(x)
                if self.dropout != None:
                    x = self.dropout(x)
        return x

class complex_relu_layer(nn.Module):
    def __init__(self, ):
        super(complex_relu_layer, self).__init__()
    
    def complex_relu(self, real, img):
        mask = 1.0*(real >= 0)
        return mask*real, mask*img

    def forward(self, real, img=None):
        # for torch nn sequential usage
        # in this case, x_real is a tuple of (real, img)
        if img == None:
            img = real[1]
            real = real[0]

        real, img = self.complex_relu(real, img)
        return real, img

class MagNet(nn.Module):
    def __init__(self, size_in, size_out, hidden_dim, K=2, nb_layers=2, enable_bias=False, droprate=False):
        """
        :param in_c: int, number of input channels.
        :param hid_c: int, number of hidden channels.
        :param K: for cheb series
        :param L_norm_real, L_norm_imag: normalized laplacian
        """
        super(MagNet, self).__init__()

        self.layers = nn.ModuleList()
        self.size_in = size_in
        self.size_out = size_out
        
        # additional linear layer to handle dimension discrepancies (if any)
        if size_in != hidden_dim:
            self.layers.append(nn.Linear(in_features=size_in, out_features=hidden_dim, bias=enable_bias))
        
        # encoder
        for _ in range(nb_layers):
            new_dim = int(hidden_dim / 2)
            self.layers.append(ChebyshevConv(K, hidden_dim, new_dim, enable_bias))
            hidden_dim = new_dim

        # decoder
        for _ in range(nb_layers):
            new_dim = int(hidden_dim * 2)
            self.layers.append(ChebyshevConv(K, hidden_dim, new_dim, enable_bias))
            hidden_dim = new_dim
        
        # additional linear layer to handle dimension discrepancies (if any) 
        if size_in != hidden_dim:
            self.layers.append(nn.Linear(in_features=hidden_dim, out_features=size_out, bias=enable_bias))

        if droprate != None:
            self.dropout = nn.Dropout(p=droprate)
        else:
            self.dropout = None
        
        self.activation = complex_relu_layer()

    def forward(self, x, L_real, L_imag):
        real = x
        imag = x
        for i in range(len(self.layers)):
            if type(self.layers[i]) == nn.Linear:
                real = self.layers[i](real)
                imag = self.layers[i](imag)
            else:
                real, imag = self.layers[i](real, L_real), self.layers[i](imag, L_imag)
                real, imag = self.activation(real, imag)
                if self.dropout != None:
                    x = self.dropout(x)
        x = torch.cat((real, imag), dim = -1)

        x = nn.Linear(in_features=2*self.size_in, out_features=self.size_out, bias=None).forward(x)
        return x
