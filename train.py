import argparse
import time

from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from model_quantize import Net
from dataset import *
import matplotlib.pyplot as plt
from evaluation import psnr
import numpy as np
# import torch.quantization
parser = argparse.ArgumentParser(description="PyTorch D3Dnet")
parser.add_argument("--save", default='./log', type=str, help="Save path")
parser.add_argument("--resume", default="", type=str, help="Resume path (default: none)")
parser.add_argument("--scale_factor", type=int, default=2, help="scale")
parser.add_argument("--train_dataset_dir", default='/Datadisk/zhangsiyu/datasets/Vimeo', type=str, help="train_dataset")
parser.add_argument("--inType", type=str, default='y', help="RGB input or y input")
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=35, help="Number of epochs to train for")
# parser.add_argument("--gpu", default=7, type=int, help="gpu ids (default: 0)")
parser.add_argument("--lr", type=float, default=4e-4, help="Learning Rate. Default=4e-4")
parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
parser.add_argument("--step", type=int, default=6, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=6")
parser.add_argument("--threads", type=int, default=2, help="Number of threads for data loader to use, Default: 1")

global opt, model
opt = parser.parse_args()
# torch.cuda.set_device(opt.gpu)
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")  #定义主卡cuda:1

def train(train_loader, scale_factor, epoch_num):

    net = Net(scale_factor).to(device)
    # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net = torch.nn.DataParallel(net, device_ids=[0,3,4,5,6])  #多卡并行，必须包含主卡
    epoch_state = 0
    loss_list = []
    psnr_list = []
    loss_epoch = []
    psnr_epoch = []

    if opt.resume:
        ckpt = torch.load(opt.resume)
        net.load_state_dict(ckpt['state_dict'])
        epoch_state = ckpt['epoch']
        loss_list = ckpt['loss']
        psnr_list = ckpt['psnr']

    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
    criterion_MSE = torch.nn.MSELoss().to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.gamma)
    for idx_epoch in range(epoch_state, epoch_num):
        for idx_iter, (LR, HR) in enumerate(train_loader):
            # print(idx_iter)
            LR, HR = Variable(LR).to(device), Variable(HR).to(device)
            SR = net(LR)

            loss = criterion_MSE(SR, HR[:, :, 3, :, :])
            loss_epoch.append(loss.detach().cpu())
            psnr_epoch.append(psnr(SR, HR[:, :, 3, :, :]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            psnr_list.append(float(np.array(psnr_epoch).mean()))
            print(time.ctime()[4:-5] + ' Epoch---%d, loss_epoch---%f, PSNR---%f' % (idx_epoch + 1, float(np.array(loss_epoch).mean()), float(np.array(psnr_epoch).mean())))
            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'loss': loss_list,
                'psnr': psnr_list,
            }, save_path=opt.save, filename='model' + str(scale_factor) + '_epoch' + str(idx_epoch + 1) + '.pth.tar')
            loss_epoch = []
            psnr_epoch = []
            valid(net)

def valid(net):
    valid_set = ValidSetLoader(opt.train_dataset_dir, scale_factor=opt.scale_factor, inType=opt.inType)
    valid_loader = DataLoader(dataset=valid_set, num_workers=opt.threads, batch_size=8, shuffle=True)
    psnr_list = []
    for idx_iter, (LR, HR) in enumerate(valid_loader):
        LR, HR = Variable(LR).to(device), Variable(HR).to(device)
        SR = net(LR)
        psnr_list.append(psnr(SR.detach(), HR[:, :, 3, :, :].detach()))
    print('valid PSNR---%f' % (float(np.array(psnr_list).mean())))

def save_checkpoint(state, save_path, filename='checkpoint.pth.tar'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(state, os.path.join(save_path,filename))

def main():
    train_set = TrainSetLoader(opt.train_dataset_dir, scale_factor=opt.scale_factor, inType=opt.inType)
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    train(train_loader, opt.scale_factor, opt.nEpochs)

if __name__ == '__main__':
    main()

