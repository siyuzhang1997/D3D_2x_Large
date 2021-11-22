import torch
torch.cuda.set_device(7)
# a = torch.zeros((2,2,4,4,4,27)).cuda()
# b = a[:,:,0,0,0,:]
# c = torch.ones_like(b).cuda()
# a[:,:,0,0,0,:] = c
# # x_offset = torch.cat([a[..., s:s + 3].contiguous().view(1, 3, 3*3) for s in range(0, 9, 3)],
# #                              dim=-1).cuda()
# # x_offset = x_offset.contiguous().view(1, 9, 9).cuda()
# result = torch.empty((2,2,12,12,12))
# for i in range(0, 4):
#     for j in range(0, 4):
#         for k in range(0,4):
#             d = a[:,:,i,j,k,:].view(2, 2, 3,3,3).permute(0, 1, 4, 2, 3)
#             result[:,:,i*3:i*3+3,j*3:j*3+3, k*3:k*3+3] = d
#
# print(1)
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from dcn.functions.deform_conv_func import DeformConvFunction

B = 1
Ci = 1
Co = 1
H = 3
W = 3
D = 2
No = 81
kernel_size = [3, 3, 3]
input = torch.ones(B, Ci, H, W, D).cuda()
# input[:,:,:,1:3,:] = 2 * input[:,:,:,1:3,:]
offset = torch.ones(B, No, H, W, D).cuda()
weight = torch.ones(Co, Ci, kernel_size[0], kernel_size[1], kernel_size[2]).cuda()
bias = None

def _get_p(offset, dtype):
    N, h, w, d = offset.size(1) // 3, offset.size(2), offset.size(3), offset.size(4)

    # (1, 3N, 1, 1, 1)
    p_n = _get_p_n(N, dtype).to(offset)
    # (1, 3N, h, w, d)
    p_0 = _get_p_0(h, w, d, N, dtype).to(offset)
    p = p_0 + p_n + offset

    return p

def _get_p_n(N, dtype):
    p_n_x, p_n_y, p_n_z = np.meshgrid(
        range(-(kernel_size[0] - 1) // 2, (kernel_size[0] - 1) // 2 + 1),
        range(-(kernel_size[1] - 1) // 2, (kernel_size[1] - 1) // 2 + 1),
        range(-(kernel_size[2] - 1) // 2, (kernel_size[2] - 1) // 2 + 1),
        indexing='ij')

    # (3N, 1) - 3 coords for each of N offsets
    # (x1, ... xN, y1, ... yN, z1, ... zN)
    p_n = np.concatenate((p_n_y.flatten(), p_n_z.flatten(), p_n_x.flatten()))
    p_n = np.reshape(p_n, (1, 3 * N, 1, 1, 1))
    p_n = torch.from_numpy(p_n).type(dtype)

    return p_n

def _get_p_0(h, w, d, N, dtype):
    p_0_x, p_0_y, p_0_z = np.meshgrid(range(1, h + 1), range(1, w+ 1), range(1, d + 1), indexing='ij')
    p_0_x = p_0_x.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
    p_0_y = p_0_y.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
    p_0_z = p_0_z.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
    p_0 = np.concatenate((p_0_x, p_0_y, p_0_z), axis=1)
    p_0 = torch.from_numpy(p_0).type(dtype)

    return p_0

def _get_x_q(x, q, N):
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

def _reshape_x_offset(x_offset, ks):
    b, c, h, w, d, N = x_offset.size()
    result = torch.empty((b, c, h * ks, w * ks, d * ks)).cuda()
    for i in range(0, h):
        for j in range(0, w):
            for k in range(0, d):
                        result[:,:,i * ks:i * ks + ks, j * ks:j *ks +ks, k * ks:k *ks + ks] = x_offset[:, :, i, j, k, :].view(b, c, ks, ks, ks)
    
    return result

dtype = offset.data.type()
N = offset.size(1) // 3

zero_padding = nn.ConstantPad3d(1, 0)
input = zero_padding(input)
p = _get_p(offset, dtype)
p = p.contiguous().permute(0, 2, 3, 4, 1)

q_sss = Variable(p.data, requires_grad=False).floor()
q_lll = q_sss + 1
# 8 neighbor points with integer coords
q_sss = torch.cat([
    torch.clamp(q_sss[..., :N], 0, input.size(2) - 1),  # h_coord
    torch.clamp(q_sss[..., N:2 * N], 0, input.size(3) - 1),  # w_coord
    torch.clamp(q_sss[..., 2 * N:], 0, input.size(4) - 1)  # d_coord
], dim=-1).long()
q_lll = torch.cat([
    torch.clamp(q_lll[..., :N], 0, input.size(2) - 1),  # h_coord
    torch.clamp(q_lll[..., N:2 * N], 0, input.size(3) - 1),  # w_coord
    torch.clamp(q_lll[..., 2 * N:], 0, input.size(4) - 1)  # d_coord
], dim=-1).long()
q_ssl = torch.cat([q_sss[..., :N], q_sss[..., N:2 * N], q_lll[..., 2 * N:]], -1)
q_sls = torch.cat([q_sss[..., :N], q_lll[..., N:2 * N], q_sss[..., 2 * N:]], -1)
q_sll = torch.cat([q_sss[..., :N], q_lll[..., N:2 * N], q_lll[..., 2 * N:]], -1)
q_lss = torch.cat([q_lll[..., :N], q_sss[..., N:2 * N], q_sss[..., 2 * N:]], -1)
q_lsl = torch.cat([q_lll[..., :N], q_sss[..., N:2 * N], q_lll[..., 2 * N:]], -1)
q_lls = torch.cat([q_lll[..., :N], q_lll[..., N:2 * N], q_sss[..., 2 * N:]], -1)

p = torch.cat([
            torch.clamp(p[..., :N], 0, input.size(2) - 1),
            torch.clamp(p[..., N:2 * N], 0, input.size(3) - 1),
            torch.clamp(p[..., 2 * N:], 0, input.size(4) - 1),
        ], dim=-1)

# trilinear kernel (b, h, w, d, N)
g_sss = (1 + (q_sss[..., :N].type_as(p) - p[..., :N])) * (
        1 + (q_sss[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                1 + (q_sss[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
g_lll = (1 - (q_lll[..., :N].type_as(p) - p[..., :N])) * (
        1 - (q_lll[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                1 - (q_lll[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
g_ssl = (1 + (q_ssl[..., :N].type_as(p) - p[..., :N])) * (
        1 + (q_ssl[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                1 - (q_ssl[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
g_sls = (1 + (q_sls[..., :N].type_as(p) - p[..., :N])) * (
        1 - (q_sls[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                1 + (q_sls[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
g_sll = (1 + (q_sll[..., :N].type_as(p) - p[..., :N])) * (
        1 - (q_sll[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                1 - (q_sll[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
g_lss = (1 - (q_lss[..., :N].type_as(p) - p[..., :N])) * (
        1 + (q_lss[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                1 + (q_lss[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
g_lsl = (1 - (q_lsl[..., :N].type_as(p) - p[..., :N])) * (
        1 + (q_lsl[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                1 - (q_lsl[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
g_lls = (1 - (q_lls[..., :N].type_as(p) - p[..., :N])) * (
        1 - (q_lls[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                1 + (q_lls[..., 2 * N:].type_as(p) - p[..., 2 * N:]))

x_offset = g_sss.unsqueeze(dim=1) * _get_x_q(input, q_sss, N) + \
           g_lll.unsqueeze(dim=1) * _get_x_q(input, q_lll, N) + \
           g_ssl.unsqueeze(dim=1) * _get_x_q(input, q_ssl, N) + \
           g_sls.unsqueeze(dim=1) * _get_x_q(input, q_sls, N) + \
           g_sll.unsqueeze(dim=1) * _get_x_q(input, q_sll, N) + \
           g_lss.unsqueeze(dim=1) * _get_x_q(input, q_lss, N) + \
           g_lsl.unsqueeze(dim=1) * _get_x_q(input, q_lsl, N) + \
           g_lls.unsqueeze(dim=1) * _get_x_q(input, q_lls, N)

x_offset = _reshape_x_offset(x_offset, kernel_size[0])
stride = (3, 3, 3)
padding = (0, 0, 0)
output = F.conv3d(
    x_offset,
    weight,
    bias,
    stride,
    padding,
)
output = output.permute(0, 1, 4, 2, 3)
# print(output)
stride_1 = (1, 1, 1)
padding_1 = (1,1,1)
dilation = (1,1,1)
DeformConvFunction.apply(input, offset,
                                        weight,
                                        bias,
                                        stride_1,
                                        padding_1,
                                        dilation,
                                        groups=1,
                                        deformable_groups=1,
                                        im2col_step=64
                         )
