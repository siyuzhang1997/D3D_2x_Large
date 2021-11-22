import argparse
import time

from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from model import Net
from dataset import *
import matplotlib.pyplot as plt
from evaluation import psnr
import numpy as np
import quantize_1 as quant_iao

parser = argparse.ArgumentParser(description="PyTorch D3Dnet")
parser.add_argument("--save", default='./log', type=str, help="Save path")
parser.add_argument("--resume", default="", type=str, help="Resume path (default: none)")
parser.add_argument("--scale_factor", type=int, default=4, help="scale")
parser.add_argument("--valid_dataset_dir", default='/Datadisk/zhangsiyu/datasets/Vimeo', type=str, help="valid_dataset")
parser.add_argument("--inType", type=str, default='y', help="RGB input or y input")
parser.add_argument("--batchSize", type=int, default=16, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=1, help="Number of epochs to train for")
parser.add_argument("--gpu", default=1, type=int, help="gpu ids (default: 0)")
parser.add_argument("--lr", type=float, default=4e-4, help="Learning Rate. Default=4e-4")
parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
parser.add_argument("--step", type=int, default=6, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=6")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")

parser.add_argument("--model", default='./log/D3Dnet.pth.tar', type=str, help="checkpoint")
parser.add_argument(
        "--bn_fuse", action="store_true", help="batch-normalization fuse" #false
    )
# bn融合校准标志位
parser.add_argument(
    "--bn_fuse_calib",
    action="store_true",
    help="batch-normalization fuse calibration",
)
# pretrained_model标志位
parser.add_argument(
    "--pretrained_model", action="store_true", help="pretrained_model"
)
parser.add_argument(
        "--qaft", action="store_true", help="quantization-aware-finetune"
    )
# ptq_observer
parser.add_argument("--ptq", action="store_false", help="post-training-quantization") #false
# ptq_control
parser.add_argument("--ptq_control", action="store_false", help="ptq control flag")   #false
# ptq_percentile
parser.add_argument(
    "--percentile", type=float, default=0.999999, help="the percentile of ptq"
)
# ptq_batch
parser.add_argument("--ptq_batch", type=int, default=5, help="the batch of ptq")


global opt, model
opt = parser.parse_args()
torch.cuda.set_device(opt.gpu)
# USE_CUDA = torch.cuda.is_available()
# device = torch.device("cuda:0" if USE_CUDA else "cpu")  #定义主卡cuda:1

def ptq(train_loader, scale_factor):

    net = Net(scale_factor)
    model = torch.load(opt.model)
    net.load_state_dict(model['state_dict'])
    print("***ori_model***\n", net)
    quant_iao.prepare(
        net,
        inplace=True,
        a_bits=8,
        w_bits=8,
        q_type=0,
        q_level=1,
        weight_observer=0,
        bn_fuse=opt.bn_fuse,
        bn_fuse_calib=opt.bn_fuse_calib,
        pretrained_model=opt.pretrained_model,
        qaft=opt.qaft,
        ptq=opt.ptq,
        percentile=opt.percentile,
    )
    print("\n***quant_model***\n", net)
    net = net.cuda()
    psnr_epoch = []
    print("ptq is doing...")
    for idx_iter, (LR, HR) in enumerate(train_loader):
        LR, HR = Variable(LR).cuda(), Variable(HR).cuda()
        SR = net(LR)

        psnr_epoch.append(psnr(SR, HR[:, :, 3, :, :]))
        if idx_iter > opt.ptq_batch:
            break
        torch.cuda.empty_cache()
        print("Batch:", idx_iter)
    aa = net.state_dict()
    for i in list(aa.keys()):
        if "min_val" in i or 'max_val' in i or 'zero' in i or 'eps' in i or 'scale' in i:
            del aa[i]
    print(aa)


def main():
    valid_set = ValidSetLoader(opt.valid_dataset_dir, scale_factor=opt.scale_factor, inType=opt.inType)
    valid_loader = DataLoader(dataset=valid_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    ptq(valid_loader, opt.scale_factor)

if __name__ == '__main__':
    main()

