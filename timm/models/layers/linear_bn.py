import torch
import torch.nn as nn
import math

from .affine import Affine


class BatchNorm(nn.Module):
    """Batch Normalization applied to Last dimension instead of 2nd dimension"""

    def __init__(self, *args, **kwargs):
        super(BatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        return x

    def fuse_bn(self):
        dim = self.bn.bias.size(-1)
        new_layer = Affine(dim)
        
        bn = self.bn
        w = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        b = bn.bias - bn.running_mean * w

        new_layer.alpha = nn.Parameter(w.view(new_layer.alpha.size()))
        new_layer.beta = nn.Parameter(b.view(new_layer.beta.size()))

        return new_layer

def fold_bn_linear(bn: BatchNorm, linear: nn.Linear):
    bn = bn.bn
    w = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    b = linear.weight @ (bn.bias - bn.running_mean * w)

    linear.weight = nn.Parameter(linear.weight * w)
    linear.bias = nn.Parameter(linear.bias + b)

    return linear


class MomentBatchNorm1dFilt(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, fifo_len=6):
        super(MomentBatchNorm1dFilt, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.fifo = []
        self.fifo_len = fifo_len

    def forward(self, x):
        x = x.permute(0, 2, 1) # -> [N,C,L] = [batch, channel, pixel/token]
        self._check_input_dim(x)

        if self.training and self.track_running_stats:
            # Calculate the variance of the input
            input_size = x.size(0) * x.size(2)
            variance_curr = ((x) ** 2).sum(dim=[0,2]) / input_size

            # Calculate the geometric mean of the last 8 variances
            variance = torch.exp(sum([torch.log(v) for v in self.fifo + [variance_curr]]) / (len(self.fifo) + 1))  # efficient way of calculating geometric mean

            if len(self.fifo) >= self.fifo_len - 1:
                self.fifo.pop(0)

            self.fifo.append(variance_curr.detach())
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * variance

        else:
            variance = self.running_var

        # Apply batch normalization
        x = (x) / (variance.unsqueeze(0).unsqueeze(2) + self.eps).sqrt()
        if self.affine:
            x = x * self.weight.unsqueeze(0).unsqueeze(2) + self.bias.unsqueeze(0).unsqueeze(2)

        x = x.permute(0, 2, 1)
        return x


class UN1dFilt(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, fifo_len=6):
        super(UN1dFilt, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.fifo = []
        self.fifo_len = fifo_len
        self.var_geo_mean = 0
        self.var_art_mean = 0
        self.test_var_stat = 0

    def forward(self, x):
        #x = x.permute(0, 2, 1) # -> [N,C,L] = [batch, channel, pixel/token]  # this permute is unnecessary and messes with parser
        self._check_input_dim(x)
        if self.training and self.track_running_stats:

            # Calculate the variance of the input
            input_size = x.size(0) * x.size(2)
            variance_curr = ((x) ** 2).sum(dim=[0,2]) / input_size   #[192,]
            variance_curr = variance_curr[:,None]                    #[192,1]

            # Calculate the statistics of the last 8/6/4/2 variances
            self.var_geo_mean = torch.exp(sum([torch.log(v) for v in self.fifo + [variance_curr]]) / (len(self.fifo) + 1))  # [192,1] efficient way of calculating geometric mean
            self.var_geo_mean = self.var_geo_mean[:,0]  # [192,]
            self.var_art_mean = torch.mean(torch.cat(self.fifo + [variance_curr], dim=1),dim=1)  # [192,]
            self.test_var_stat = torch.var(torch.sqrt(torch.cat(self.fifo + [variance_curr], dim=1)), dim=1)  # [192,]

            # check E[sigma**2] - PI[sigma**2] <= M*var[sigma] - for all entires in the 192 tensor!
            if torch.all(self.var_art_mean - self.var_geo_mean <= self.fifo_len*self.test_var_stat):
                variance = self.var_geo_mean
            else:
                variance =  variance_curr[:,0]

            if len(self.fifo) >= self.fifo_len - 1:
                self.fifo.pop(0)

            self.fifo.append(variance_curr.detach())
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * variance

        else:
            variance = self.running_var

        # Apply batch normalization
        '''
        # unnecessary and messes with parser
        x = (x) / (variance.unsqueeze(0).unsqueeze(2) + self.eps).sqrt()
        if self.affine:
            x = x * self.weight.unsqueeze(0).unsqueeze(2) + self.bias.unsqueeze(0).unsqueeze(2)
        x = x.permute(0, 2, 1)
        '''

        x = (x) / (variance.unsqueeze(0).unsqueeze(1) + self.eps).sqrt()
        if self.affine:
            x = x * self.weight.unsqueeze(0).unsqueeze(1) + self.bias.unsqueeze(0).unsqueeze(1)
        return x


class UN1dFilt_wo_permute(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, fifo_len=6):
        super(UN1dFilt_wo_permute, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.fifo = []
        self.fifo_len = fifo_len
        self.var_geo_mean = 0
        self.var_art_mean = 0
        self.test_var_stat = 0

    def forward(self, x):
        self._check_input_dim(x)
        if self.training and self.track_running_stats:

            # Calculate the variance of the input
            input_size = x.size(0) * x.size(1)
            variance_curr = ((x) ** 2).sum(dim=[0,1]) / input_size   #[192,]
            variance_curr = variance_curr[:,None]                    #[192,1]

            # Calculate the statistics of the last 8/6/4/2 variances
            self.var_geo_mean = torch.exp(sum([torch.log(v) for v in self.fifo + [variance_curr]]) / (len(self.fifo) + 1))  # [192,1] efficient way of calculating geometric mean
            self.var_geo_mean = self.var_geo_mean[:,0]  # [192,]
            self.var_art_mean = torch.mean(torch.cat(self.fifo + [variance_curr], dim=1),dim=1)  # [192,]
            self.test_var_stat = torch.var(torch.sqrt(torch.cat(self.fifo + [variance_curr], dim=1)), dim=1)  # [192,]

            # check E[sigma**2] - PI[sigma**2] <= M*var[sigma] - for all entires in the 192 tensor!
            if torch.all(self.var_art_mean - self.var_geo_mean <= self.fifo_len*self.test_var_stat):
                variance = self.var_geo_mean
            else:
                variance =  variance_curr[:,0]

            if len(self.fifo) >= self.fifo_len - 1:
                self.fifo.pop(0)

            self.fifo.append(variance_curr.detach())
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * variance

        else:
            variance = self.running_var

        x = (x) / (variance.unsqueeze(0).unsqueeze(1) + self.eps).sqrt()
        if self.affine:
            x = x * self.weight.unsqueeze(0).unsqueeze(1) + self.bias.unsqueeze(0).unsqueeze(1)
        return x