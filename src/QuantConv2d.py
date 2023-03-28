import torch
import torch.nn as nn
import cu_gemm_quant
import Config as cfg
#from Config import X_FP,W_FP,SUM_FP
import matplotlib.pyplot as plt
import our_utils as fpQuant
import os
from time import time
import copy

def plot_activations(activation):
    fig = plt.figure(figsize = (12,8))
    plt.hist(activation.cpu().flatten(), bins=100)
    plt.title('activations')
    fig.savefig(os.path.join("./hists/", 'Activations_hist_{}.png'.format(time())))
    plt.close(fig)

def sign_removal(x, input_dic, percantge):
    """
    sign removal when most of the values are positive.
    percantge of the lowest values
    """
    x_calc = x.flatten()
    k = int(percantge * x_calc.numel() / 100)
    topk = torch.topk(x_calc,k,largest=False)
    threshold = torch.max(topk[0]).item()
    if threshold > 0:
        input_dic['exponent'] = 0
        x[x<0] = 0
    return x, input_dic
    
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class UnfoldConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(UnfoldConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, groups=groups,
                                           bias=bias, padding_mode=padding_mode)

        # Registering buffers to be saved when calling torch.save
        self.register_buffer('tracked_n', torch.zeros(1))
        self.register_buffer('max_mean', torch.zeros(1))
        self.register_buffer('min_mean', torch.zeros(1))

        # Even if in training mode, the user can disable gathering the tensor min-max values
        self._disable_min_max_update = False

        # Quantization variables
        self._quantize = False
        self._x_bits = 8
        self._w_bits = 8

        # Custom kernel variables
        self._is_round = None
        self._shift_opt_x = None
        self._shift_opt_w = None

    def _reset_stats(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                self._reset_stats(v)
            else:
                d[k] = 0

    def reset_stats(self):
        self._reset_stats(self.stats)

    def forward(self, x):
        #percantge = 46
        x_input_order_bits = copy.deepcopy(cfg.X_FP)
        w_input_order_bits = copy.deepcopy(cfg.W_FP)
        sum_input_order_bits = copy.deepcopy(cfg.SUM_FP)
        ieee_x = False
        ieee_w = False
        ieee_sum = False
        max_bias_x = True
        max_bias_w = True
        max_bias_sum = False
        w = self.weight
        b = self.bias
        # Prepare activations, weights, and bias
        if self._quantize:

            # Gather statistics during training
            if self.training and not self._disable_min_max_update:
                tracked_n_old = self.tracked_n.clone()
                self.tracked_n += x.size(0)
            # if there are no negative elements dont use the sign bit ad use max bias
            #x_s, x_input_order_bits = sign_removal(x, x_input_order_bits, percantge)
            
            
            x_q = fpQuant.closestQuant(fp_bits=self._x_bits, w=x, input_order_bits=x_input_order_bits, ieee=ieee_x, max_bias = max_bias_x)
            #x_q=x
            #plot_activations(x_q)


            # Weights quantization
            #w_s, w_input_order_bits = sign_removal(w, w_input_order_bits, percantge)
            w_q = fpQuant.closestQuant(fp_bits=self._w_bits, w=w , input_order_bits=w_input_order_bits, ieee=ieee_w, max_bias = max_bias_w)
            #w_q=self.weight


            # Bias quantization
            if self.bias is None:
                bias_fp = None
            else:
                #b_s, w_input_order_bits = sign_removal(b, w_input_order_bits, percantge)
                bias_fp = fpQuant.closestQuant(fp_bits=self._w_bits, w=b , input_order_bits=w_input_order_bits, ieee=ieee_w, max_bias = max_bias_w)
                #bias_fp=self.bias

        else:
            # The single scalar movement to CUDA may be bad for performance
            x_q = x
            w_q= self.weight
            bias_fp = self.bias

        out = nn.functional.conv2d(x_q,
                                   w_q,
                                   bias=bias_fp,
                                   stride=(self.stride[0], self.stride[1]),
                                   padding=(self.padding[0], self.padding[1]), groups=self.groups)
        #out_s, sum_input_order_bits = sign_removal(out, sum_input_order_bits, percantge)
        out_q = fpQuant.closestQuant(fp_bits=self._x_bits, w=out , input_order_bits=sum_input_order_bits, ieee=ieee_sum, max_bias = max_bias_sum) #TODO same size as x
        return out_q

    def get_status_arr(self):
        key, val = [], []

        key.extend(['quant', 'x_b', 'w_b'])
        val.extend([self._quantize, self._x_bits, self._w_bits])

        key.extend(['sparq_x', 'sparq_w'])
        val.extend([self._sparq_x, self._sparq_w])

        key.extend(['is_round'])
        val.extend([self._is_round]) if self._sparq_x or self._sparq_w else val.extend(['-', '-'])

        key.extend(['shift_opt_x'])
        val.extend([self._shift_opt_x]) if self._sparq_x else val.extend(['-'])

        key.extend(['shift_opt_w'])
        val.extend([self._shift_opt_w]) if self._sparq_w else val.extend(['-'])

        key.extend(['grp_sz_x'])
        val.extend([self._group_sz_x]) if self._sparq_x else val.extend(['-', '-'])

        key.extend(['grp_sz_w'])
        val.extend([self._group_sz_w]) if self._sparq_w else val.extend(['-', '-'])

        return key, val

    @staticmethod
    def _uniform_quantization(x, x_max, bits):
        N = 2 ** bits
        delta = x_max / (N - 1)
        x_int = RoundSTE.apply(x / delta)
        x_q = torch.clamp(x_int, 0, N - 1)
        return x_q, delta

    @staticmethod
    def _uniform_symmetric_quantization_per_filter(x, x_min, x_max, bits):
        N = 2 ** bits
        delta = torch.where(x_min.abs() > x_max.abs(), x_min.abs(), x_max.abs()) * 2 / (N - 1)
        x_int = RoundSTE.apply(x / delta[:, None, None, None].expand_as(x))
        x_q = torch.clamp(x_int, -N / 2, N / 2 - 1)
        return x_q, delta

    @staticmethod
    def _uniform_symmetric_quantization(x, x_min, x_max, bits):
        N = 2 ** bits
        delta = max(abs(x_min), abs(x_max)) * 2 / (N - 1)
        x_int = RoundSTE.apply(x / delta)
        x_q = torch.clamp(x_int, -N / 2, N / 2 - 1)
        return x_q, delta
