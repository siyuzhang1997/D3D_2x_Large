#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np
import torch
import math
from torch import nn
from torch.nn import init
from torch.nn.modules.utils import _triple
from torch.autograd import Variable
from dcn.functions.deform_conv_func import DeformConvFunction
import torch.nn.functional as F

class DeformConv(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True):
        super(DeformConv, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels {} must be divisible by groups {}'.format(in_channels, groups))
        if out_channels % groups != 0:
            raise ValueError('out_channels {} must be divisible by groups {}'.format(out_channels, groups))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.im2col_step = im2col_step
        self.use_bias = bias
        
        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels//groups, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
        if not self.use_bias:
            self.bias.requires_grad = False

    def reset_parameters(self):
        n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input, offset):
        assert 3 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] == \
            offset.shape[1]
        return DeformConvFunction.apply(input, offset,
                                                   self.weight,
                                                   self.bias,
                                                   self.stride,
                                                   self.padding,
                                                   self.dilation,
                                                   self.groups,
                                                   self.deformable_groups,
                                                   self.im2col_step)

_DeformConv = DeformConvFunction.apply

class DeformConvPack(DeformConv):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True, lr_mult=0.1):
        super(DeformConvPack, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, groups, deformable_groups, im2col_step, bias)


        out_channels = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        self.conv_offset = nn.Conv3d(self.in_channels,
                                          out_channels,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)
        self.conv_offset.lr_mult = lr_mult
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, input):
        offset = self.conv_offset(input)
        return DeformConvFunction.apply(input, offset,
                                          self.weight, 
                                          self.bias, 
                                          self.stride, 
                                          self.padding, 
                                          self.dilation, 
                                          self.groups,
                                          self.deformable_groups,
                                          self.im2col_step)


class DeformConv_d(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dimension='THW', dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True):
        super(DeformConv_d, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels {} must be divisible by groups {}'.format(in_channels, groups))
        if out_channels % groups != 0:
            raise ValueError('out_channels {} must be divisible by groups {}'.format(out_channels, groups))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.dimension = dimension
        self.length = len(dimension)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.im2col_step = im2col_step
        self.use_bias = bias

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
        if not self.use_bias:
            self.bias.requires_grad = False

    def reset_parameters(self):
        n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input, temp):
        dimension_T = 'T' in self.dimension
        dimension_H = 'H' in self.dimension
        dimension_W = 'W' in self.dimension
        b, c, t, h, w = temp.shape
        if self.length == 2:
            temp1 = temp.clone()[:, 0:81 - c, :, :, :]
            offset = torch.cat((temp.clone(), temp1), dim=1)
            if dimension_T == False:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = 0  # T
                    offset[:, i * 3 + 1, :, :, :] = temp[:, i * 2, :, :, :]
                    offset[:, i * 3 + 2, :, :, :] = temp[:, i * 2 + 1, :, :, :]
            if dimension_H == False:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = temp[:, i * 2, :, :, :]
                    offset[:, i * 3 + 1, :, :, :] = 0  # H
                    offset[:, i * 3 + 2, :, :, :] = temp[:, i * 2 + 1, :, :, :]
            if dimension_W == False:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = temp[:, i * 2, :, :, :]
                    offset[:, i * 3 + 1, :, :, :] = temp[:, i * 2 + 1, :, :, :]
                    offset[:, i * 3 + 2, :, :, :] = 0  # W

        if self.length == 1:
            temp1 = temp.clone()
            offset = torch.cat((temp.clone(), temp1, temp1), dim=1)
            if dimension_T == True:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = temp[:, i, :, :, :]  # T
                    offset[:, i * 3 + 1, :, :, :] = 0
                    offset[:, i * 3 + 2, :, :, :] = 0
            if dimension_H == True:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = 0
                    offset[:, i * 3 + 1, :, :, :] = temp[:, i, :, :, :]  # H
                    offset[:, i * 3 + 2, :, :, :] = 0
            if dimension_W == True:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = 0
                    offset[:, i * 3 + 1, :, :, :] = 0
                    offset[:, i * 3 + 2, :, :, :] = temp[:, i, :, :, :]  # W

        return DeformConvFunction.apply(input, offset,
                                        self.weight,
                                        self.bias,
                                        self.stride,
                                        self.padding,
                                        self.dilation,
                                        self.groups,
                                        self.deformable_groups,
                                        self.im2col_step)


_DeformConv = DeformConvFunction.apply


class DeformConvPack_d(DeformConv_d):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dimension='THW',
                 dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True, lr_mult=0.1):
        super(DeformConvPack_d, self).__init__(in_channels, out_channels,
                                             kernel_size, stride, padding, dimension, dilation, groups, deformable_groups,
                                             im2col_step, bias)
        self.dimension = dimension
        self.length = len(dimension)
        out_channels = self.deformable_groups * self.length * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        self.conv_offset = nn.Conv3d(self.in_channels,
                                     out_channels,
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     padding=self.padding,
                                     bias=True)
        self.zero_padding = nn.ConstantPad3d(self.padding[0], 0)
        self.conv_offset.lr_mult = lr_mult
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, input):

        temp = self.conv_offset(input)
        dimension_T = 'T' in self.dimension
        dimension_H = 'H' in self.dimension
        dimension_W = 'W' in self.dimension
        b, c, t, h, w = temp.shape
        if self.length == 2:
            temp1 = temp.clone()[:, 0:81 - c, :, :, :]
            offset = torch.cat((temp.clone(), temp1), dim=1)
            if dimension_T == False:
                for i in range(  # THW
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = 0  # T
                    offset[:, i * 3 + 1, :, :, :] = temp[:, i * 2, :, :, :]
                    offset[:, i * 3 + 2, :, :, :] = temp[:, i * 2 + 1, :, :, :]
            if dimension_H == False:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = temp[:, i * 2, :, :, :]
                    offset[:, i * 3 + 1, :, :, :] = 0  # H
                    offset[:, i * 3 + 2, :, :, :] = temp[:, i * 2 + 1, :, :, :]
            if dimension_W == False:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = temp[:, i * 2, :, :, :]
                    offset[:, i * 3 + 1, :, :, :] = temp[:, i * 2 + 1, :, :, :]
                    offset[:, i * 3 + 2, :, :, :] = 0  # W

        if self.length == 1:
            temp1 = temp.clone()
            offset = torch.cat((temp.clone(), temp1, temp1), dim=1)
            if dimension_T == True:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = temp[:, i, :, :, :]  # T
                    offset[:, i * 3 + 1, :, :, :] = 0
                    offset[:, i * 3 + 2, :, :, :] = 0
            if dimension_H == True:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = 0
                    offset[:, i * 3 + 1, :, :, :] = temp[:, i, :, :, :]  # H
                    offset[:, i * 3 + 2, :, :, :] = 0
            if dimension_W == True:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = 0
                    offset[:, i * 3 + 1, :, :, :] = 0
                    offset[:, i * 3 + 2, :, :, :] = temp[:, i, :, :, :]  # W

        return DeformConvFunction.apply(input, offset,  # input(1,64,7,144,180) offset(1,27*3,7,144,180)
                                        self.weight,
                                        self.bias,
                                        self.stride,
                                        self.padding,
                                        self.dilation,
                                        self.groups,
                                        self.deformable_groups,
                                        self.im2col_step)
    # def forward(self, input):
    #     temp = self.conv_offset(input)
    #     # print(temp.max())
    #     # print(temp.min())
    #     dimension_T = 'T' in self.dimension
    #     b, c, t, h, w = temp.shape
    #     if self.length == 2:
    #         temp1 = temp.clone()[:, 0:81 - c, :, :, :]
    #         offset = torch.cat((temp.clone(), temp1), dim=1)
    #         if dimension_T == False:
    #             for i in range(   #HWT
    #                     self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
    #                 offset[:, i, :, :, :] = temp[:, i * 2, :, :, :]
    #                 offset[:, i+27, :, :, :] = temp[:, i * 2 + 1, :, :, :]
    #                 offset[:, i+54, :, :, :] = 0  # W
    #
    #     offset = offset.contiguous().permute(0, 1, 3, 4, 2)
    #     input = input.contiguous().permute(0, 1, 3, 4, 2)
    #     dtype = offset.data.type()
    #     N = offset.size(1) // 3
    #
    #     if self.padding:
    #         input = self.zero_padding(input)
    #
    #     # (b, 3N, h, w, d)
    #     p = self._get_p(offset, dtype)
    #     # p = p[:, :, ::self.stride[0], ::self.stride[1], ::self.stride[2]]
    #
    #     # (b, h, w, d, 3N), N == ks ** 3, 3N - 3 coords for each point on the activation map
    #     p = p.contiguous().permute(0, 2, 3, 4, 1)  # 5D array
    #
    #     q_sss = Variable(p.data, requires_grad=False).floor()  # point with all smaller coords
    #     #         q_sss = p.data.floor() - same? / torch.Tensor(p.data).floor()
    #     q_lll = q_sss + 1  # all larger coords
    #
    #     # 8 neighbor points with integer coords
    #     q_sss = torch.cat([
    #         torch.clamp(q_sss[..., :N], 0, input.size(2) - 1),  # h_coord
    #         torch.clamp(q_sss[..., N:2 * N], 0, input.size(3) - 1),  # w_coord
    #         torch.clamp(q_sss[..., 2 * N:], 0, input.size(4) - 1)  # d_coord
    #     ], dim=-1).long()
    #     q_lll = torch.cat([
    #         torch.clamp(q_lll[..., :N], 0, input.size(2) - 1),  # h_coord
    #         torch.clamp(q_lll[..., N:2 * N], 0, input.size(3) - 1),  # w_coord
    #         torch.clamp(q_lll[..., 2 * N:], 0, input.size(4) - 1)  # d_coord
    #     ], dim=-1).long()
    #     q_ssl = torch.cat([q_sss[..., :N], q_sss[..., N:2 * N], q_lll[..., 2 * N:]], -1)
    #     q_sls = torch.cat([q_sss[..., :N], q_lll[..., N:2 * N], q_sss[..., 2 * N:]], -1)
    #     q_sll = torch.cat([q_sss[..., :N], q_lll[..., N:2 * N], q_lll[..., 2 * N:]], -1)
    #     q_lss = torch.cat([q_lll[..., :N], q_sss[..., N:2 * N], q_sss[..., 2 * N:]], -1)
    #     q_lsl = torch.cat([q_lll[..., :N], q_sss[..., N:2 * N], q_lll[..., 2 * N:]], -1)
    #     q_lls = torch.cat([q_lll[..., :N], q_lll[..., N:2 * N], q_sss[..., 2 * N:]], -1)
    #
    #     # (b, h, w, d, N)
    #     # mask = torch.cat([
    #     #     p[..., :N].lt(self.padding[0]) + p[..., :N].gt(input.size(2) - 1 - self.padding[0]),
    #     #     p[..., N:2 * N].lt(self.padding[1]) + p[..., N:2 * N].gt(input.size(3) - 1 - self.padding[1]),
    #     #     p[..., 2 * N:].lt(self.padding[2]) + p[..., 2 * N:].gt(input.size(4) - 1 - self.padding[2]),
    #     # ], dim=-1).type_as(p)
    #     # mask = mask.detach()
    #     # floor_p = p - (p - torch.floor(p))  # все еще непонятно, что тут происходит за wtf
    #     # p = p * (1 - mask) + floor_p * mask
    #
    #     p = torch.cat([
    #         torch.clamp(p[..., :N], 0, input.size(2) - 1),
    #         torch.clamp(p[..., N:2 * N], 0, input.size(3) - 1),
    #         torch.clamp(p[..., 2 * N:], 0, input.size(4) - 1),
    #     ], dim=-1)
    #
    #     # trilinear kernel (b, h, w, d, N)
    #     g_sss = (1 + (q_sss[..., :N].type_as(p) - p[..., :N])) * (
    #             1 + (q_sss[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
    #                     1 + (q_sss[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
    #     g_lll = (1 - (q_lll[..., :N].type_as(p) - p[..., :N])) * (
    #             1 - (q_lll[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
    #                     1 - (q_lll[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
    #     g_ssl = (1 + (q_ssl[..., :N].type_as(p) - p[..., :N])) * (
    #             1 + (q_ssl[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
    #                     1 - (q_ssl[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
    #     g_sls = (1 + (q_sls[..., :N].type_as(p) - p[..., :N])) * (
    #             1 - (q_sls[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
    #                     1 + (q_sls[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
    #     g_sll = (1 + (q_sll[..., :N].type_as(p) - p[..., :N])) * (
    #             1 - (q_sll[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
    #                     1 - (q_sll[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
    #     g_lss = (1 - (q_lss[..., :N].type_as(p) - p[..., :N])) * (
    #             1 + (q_lss[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
    #                     1 + (q_lss[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
    #     g_lsl = (1 - (q_lsl[..., :N].type_as(p) - p[..., :N])) * (
    #             1 + (q_lsl[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
    #                     1 - (q_lsl[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
    #     g_lls = (1 - (q_lls[..., :N].type_as(p) - p[..., :N])) * (
    #             1 - (q_lls[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
    #                     1 + (q_lls[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
    #
    #     x_offset = g_sss.unsqueeze(dim=1) * self._get_x_q(input, q_sss, N) + \
    #                g_lll.unsqueeze(dim=1) * self._get_x_q(input, q_lll, N) + \
    #                g_ssl.unsqueeze(dim=1) * self._get_x_q(input, q_ssl, N) + \
    #                g_sls.unsqueeze(dim=1) * self._get_x_q(input, q_sls, N) + \
    #                g_sll.unsqueeze(dim=1) * self._get_x_q(input, q_sll, N) + \
    #                g_lss.unsqueeze(dim=1) * self._get_x_q(input, q_lss, N) + \
    #                g_lsl.unsqueeze(dim=1) * self._get_x_q(input, q_lsl, N) + \
    #                g_lls.unsqueeze(dim=1) * self._get_x_q(input, q_lls, N)
    #
    #     x_offset = self._reshape_x_offset(x_offset, self.kernel_size[0])
    #     stride = (3, 3, 3)
    #     padding = (0, 0, 0)
    #     output = F.conv3d(
    #         x_offset,
    #         self.weight,
    #         self.bias,
    #         stride,
    #         padding,
    #         self.dilation,
    #         self.groups,
    #     )
    #     return output
    # def forward(self, input):
    #     temp = self.conv_offset(input)
    #     # print(temp.max())
    #     # print(temp.min())
    #     dimension_T = 'T' in self.dimension
    #     b, c, t, h, w = temp.shape
    #     if self.length == 2:
    #         temp1 = temp.clone()[:, 0:81 - c, :, :, :]
    #         offset = torch.cat((temp.clone(), temp1), dim=1)
    #         if dimension_T == False:
    #             for i in range(   #HWT
    #                     self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
    #                 offset[:, i, :, :, :] = temp[:, i * 2, :, :, :]
    #                 offset[:, i+27, :, :, :] = temp[:, i * 2 + 1, :, :, :]
    #                 offset[:, i+54, :, :, :] = 0  # W
    #
    #     offset = offset.contiguous().permute(0, 1, 3, 4, 2)
    #     input = input.contiguous().permute(0, 1, 3, 4, 2)
    #     dtype = offset.data.type()
    #     N = offset.size(1) // 3
    #
    #     if self.padding:
    #         input = self.zero_padding(input)
    #
    #     # (b, 3N, h, w, d)
    #     p = self._get_p(offset, dtype)
    #     # p = p[:, :, ::self.stride[0], ::self.stride[1], ::self.stride[2]]
    #
    #     # (b, h, w, d, 3N), N == ks ** 3, 3N - 3 coords for each point on the activation map
    #     p = p.contiguous().permute(0, 2, 3, 4, 1)  # 5D array
    #
    #     q_sss = Variable(p.data, requires_grad=False).floor()  # point with all smaller coords
    #     #         q_sss = p.data.floor() - same? / torch.Tensor(p.data).floor()
    #     q_lll = q_sss + 1  # all larger coords
    #
    #     # 8 neighbor points with integer coords
    #     q_sss = torch.cat([
    #         torch.clamp(q_sss[..., :N], 0, input.size(2) - 1),  # h_coord
    #         torch.clamp(q_sss[..., N:2 * N], 0, input.size(3) - 1),  # w_coord
    #         torch.clamp(q_sss[..., 2 * N:], 0, input.size(4) - 1)  # d_coord
    #     ], dim=-1).long()
    #     q_lll = torch.cat([
    #         torch.clamp(q_lll[..., :N], 0, input.size(2) - 1),  # h_coord
    #         torch.clamp(q_lll[..., N:2 * N], 0, input.size(3) - 1),  # w_coord
    #         torch.clamp(q_lll[..., 2 * N:], 0, input.size(4) - 1)  # d_coord
    #     ], dim=-1).long()
    #     q_ssl = torch.cat([q_sss[..., :N], q_sss[..., N:2 * N], q_lll[..., 2 * N:]], -1)
    #     q_sls = torch.cat([q_sss[..., :N], q_lll[..., N:2 * N], q_sss[..., 2 * N:]], -1)
    #     q_sll = torch.cat([q_sss[..., :N], q_lll[..., N:2 * N], q_lll[..., 2 * N:]], -1)
    #     q_lss = torch.cat([q_lll[..., :N], q_sss[..., N:2 * N], q_sss[..., 2 * N:]], -1)
    #     q_lsl = torch.cat([q_lll[..., :N], q_sss[..., N:2 * N], q_lll[..., 2 * N:]], -1)
    #     q_lls = torch.cat([q_lll[..., :N], q_lll[..., N:2 * N], q_sss[..., 2 * N:]], -1)
    #
    #     # (b, h, w, d, N)
    #     # mask = torch.cat([
    #     #     p[..., :N].lt(self.padding[0]) + p[..., :N].gt(input.size(2) - 1 - self.padding[0]),
    #     #     p[..., N:2 * N].lt(self.padding[1]) + p[..., N:2 * N].gt(input.size(3) - 1 - self.padding[1]),
    #     #     p[..., 2 * N:].lt(self.padding[2]) + p[..., 2 * N:].gt(input.size(4) - 1 - self.padding[2]),
    #     # ], dim=-1).type_as(p)
    #     # mask = mask.detach()
    #     # floor_p = p - (p - torch.floor(p))  # все еще непонятно, что тут происходит за wtf
    #     # p = p * (1 - mask) + floor_p * mask
    #
    #     p = torch.cat([
    #         torch.clamp(p[..., :N], 0, input.size(2) - 1),
    #         torch.clamp(p[..., N:2 * N], 0, input.size(3) - 1),
    #         torch.clamp(p[..., 2 * N:], 0, input.size(4) - 1),
    #     ], dim=-1)
    #
    #     # trilinear kernel (b, h, w, d, N)
    #     g_sss = (1 + (q_sss[..., :N].type_as(p) - p[..., :N])) * (
    #             1 + (q_sss[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
    #                     1 + (q_sss[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
    #     g_lll = (1 - (q_lll[..., :N].type_as(p) - p[..., :N])) * (
    #             1 - (q_lll[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
    #                     1 - (q_lll[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
    #     g_ssl = (1 + (q_ssl[..., :N].type_as(p) - p[..., :N])) * (
    #             1 + (q_ssl[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
    #                     1 - (q_ssl[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
    #     g_sls = (1 + (q_sls[..., :N].type_as(p) - p[..., :N])) * (
    #             1 - (q_sls[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
    #                     1 + (q_sls[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
    #     g_sll = (1 + (q_sll[..., :N].type_as(p) - p[..., :N])) * (
    #             1 - (q_sll[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
    #                     1 - (q_sll[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
    #     g_lss = (1 - (q_lss[..., :N].type_as(p) - p[..., :N])) * (
    #             1 + (q_lss[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
    #                     1 + (q_lss[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
    #     g_lsl = (1 - (q_lsl[..., :N].type_as(p) - p[..., :N])) * (
    #             1 + (q_lsl[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
    #                     1 - (q_lsl[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
    #     g_lls = (1 - (q_lls[..., :N].type_as(p) - p[..., :N])) * (
    #             1 - (q_lls[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
    #                     1 + (q_lls[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
    #
    #     x_offset = g_sss.unsqueeze(dim=1) * self._get_x_q(input, q_sss, N) + \
    #                g_lll.unsqueeze(dim=1) * self._get_x_q(input, q_lll, N) + \
    #                g_ssl.unsqueeze(dim=1) * self._get_x_q(input, q_ssl, N) + \
    #                g_sls.unsqueeze(dim=1) * self._get_x_q(input, q_sls, N) + \
    #                g_sll.unsqueeze(dim=1) * self._get_x_q(input, q_sll, N) + \
    #                g_lss.unsqueeze(dim=1) * self._get_x_q(input, q_lss, N) + \
    #                g_lsl.unsqueeze(dim=1) * self._get_x_q(input, q_lsl, N) + \
    #                g_lls.unsqueeze(dim=1) * self._get_x_q(input, q_lls, N)
    #
    #     x_offset = self._reshape_x_offset(x_offset, self.kernel_size[0])
    #     stride = (3, 3, 3)
    #     padding = (0, 0, 0)
    #     output = F.conv3d(
    #         x_offset,
    #         self.weight,
    #         self.bias,
    #         stride,
    #         padding,
    #         self.dilation,
    #         self.groups,
    #     )
    #     return output

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y, p_n_z = np.meshgrid(
            range(-(self.kernel_size[0] - 1) // 2, (self.kernel_size[0] - 1) // 2 + 1),
            range(-(self.kernel_size[1] - 1) // 2, (self.kernel_size[1] - 1) // 2 + 1),
            range(-(self.kernel_size[2] - 1) // 2, (self.kernel_size[2] - 1) // 2 + 1),
            indexing='ij')

        # (3N, 1) - 3 coords for each of N offsets
        # (x1, ... xN, y1, ... yN, z1, ... zN)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten(), p_n_z.flatten()))
        p_n = np.reshape(p_n, (1, 3 * N, 1, 1, 1))
        p_n = torch.from_numpy(p_n).type(dtype)

        return p_n

    @staticmethod
    def _get_p_0(h, w, d, N, dtype):
        p_0_x, p_0_y, p_0_z = np.meshgrid(range(1, h + 1), range(1, w+ 1), range(1, d + 1), indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0_z = p_0_z.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y, p_0_z), axis=1)
        p_0 = torch.from_numpy(p_0).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w, d = offset.size(1) // 3, offset.size(2), offset.size(3), offset.size(4)

        # (1, 3N, 1, 1, 1)
        p_n = self._get_p_n(N, dtype).to(offset)
        # (1, 3N, h, w, d)
        p_0 = self._get_p_0(h, w, d, N, dtype).to(offset)
        p = p_0 + p_n + offset

        return p

    def _get_x_q(self, x, q, N):
        b, h, w, d, _ = q.size()

        #           (0, 1, 2, 3, 4)
        # x.size == (b, c, h, w, d)
        padded_w = x.size(3)
        padded_d = x.size(4)
        c = x.size(1)
        # (b, c, h*w*d)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, d, N)
        # offset_x * w * d + offset_y * d + offset_z
        index = q[..., :N] * padded_w * padded_d + q[..., N:2 * N] * padded_d + q[..., 2 * N:]
        # (b, c, h*w*d*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, d, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, d, N = x_offset.size()
        result = torch.empty((b, c, h * ks, w * ks, d * ks)).cuda()
        for k in range(0, d):
            for j in range(0, w):
                for i in range(0, h):
                        result[:,:,i * ks:i * ks + ks, j * ks:j *ks +ks, k * ks:k *ks + ks] = x_offset[:, :, i, j, k, :].view(b, c, ks, ks, ks)
        x_offset = result.permute(0, 1, 4, 2, 3)

        # x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w*ks, d) for s in range(0, N, ks)],
        #                      dim=-3)
        # x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks, d * ks)
        # x_offset = x_offset.permute(0, 1, 4, 2, 3)


        return x_offset

