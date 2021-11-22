from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import *

from model_quantize import *
from evaluation import psnr2, ssim
import numpy as np
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import argparse, os

parser = argparse.ArgumentParser(description="PyTorch D3Dnet")
parser.add_argument("--scale_factor", type=int, default=4, help="scale")
parser.add_argument("--test_dataset_dir", default='./data', type=str, help="test_dataset dir")
parser.add_argument("--test_dataset_dir_Vimeo", default='/Datadisk/zhangsiyu/datasets', type=str, help="test_dataset dir")
# parser.add_argument("--model", default='./log/D3Dnet.pth.tar', type=str, help="checkpoint")
parser.add_argument("--model", default='./log/D3Dnet.pth.tar', type=str, help="checkpoint")
parser.add_argument("--inType", type=str, default='y', help="RGB input or y input")
parser.add_argument("--batchSize", type=int, default=1, help="Test batch size")
parser.add_argument("--gpu", type=int, default=4, help="Test batch size")
# parser.add_argument("--datasets", type=str, default=['Vid4','SPMC-11','Vimeo'], help="Test batch size")
parser.add_argument("--datasets", type=str, default=['Vid4'], help="Test batch size")
global opt, model
opt = parser.parse_args()
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

def demo_test(net, test_loader, scale_factor, dataset_name, video_name):
    PSNR_list = []
    SSIM_list = []
    with torch.no_grad():
        for idx_iter, (LR, HR, SR_buicbic) in enumerate(test_loader):
            LR, HR = Variable(LR).to(device), Variable(HR).to(device)
            SR = net(LR)   #LR->SR，同时SR是单通道，应该是Y通道
            SR = torch.clamp(SR, 0, 1) #0和1的截断函数

            PSNR_list.append(psnr2(SR, HR[:, :, 3, :, :])) #评价指标仅仅在Y通道
            SSIM_list.append(ssim(SR, HR[:, :, 3, :, :]))

            if not os.path.exists('results/' + dataset + '/' + video_name):
                os.makedirs('./results/' + dataset + '/' + video_name)
            ## save y images
            # SR_img = transforms.ToPILImage()(SR[0, :, :, :].cpu())
            # SR_img.save('results/' + dataset_name + '/' + video_name + '/sr_y_' + str(idx_iter+1).rjust(2, '0') + '.png')

            ## save rgb images
            SR_buicbic[:, 0, :, :] = SR[:, 0, :, :].cpu()   #用计算的Y通道的SR结果替换对LR插值的结果
            SR_rgb = (ycbcr2rgb(SR_buicbic[0,:,:,:].permute(2,1,0))).permute(2,1,0)
            SR_rgb = torch.clamp(SR_rgb, 0, 1)
            SR_img = transforms.ToPILImage()(SR_rgb)
            SR_img.save('results/' + dataset_name + '/' + video_name + '/sr_rgb_' + str(idx_iter+1).rjust(2, '0') + '.png')

        PSNR_mean = float(torch.cat(PSNR_list, 0)[2:-2].data.cpu().mean())
        SSIM_mean = float(torch.cat(SSIM_list, 0)[2:-2].data.cpu().mean())
        print(video_name + ' psnr: '+ str(PSNR_mean) + ' ssim: ' + str(SSIM_mean))
        return PSNR_mean, SSIM_mean

def main(dataset_name):
    net = Net(opt.scale_factor).to(device)
    # net = torch.nn.DataParallel(net, device_ids=[0, 1, 2, 3])
    model = torch.load(opt.model)

    net.load_state_dict(model['state_dict'])
    print(net)
    PSNR_dataset = []
    SSIM_dataset = []

    if dataset_name == 'Vid4' or dataset_name == 'SPMC-11':
        video_list = os.listdir(opt.test_dataset_dir + '/' + dataset_name)
        for i in range(0, len(video_list)):
            video_name = video_list[i]
            test_set = TestSetLoader(opt.test_dataset_dir + '/' + dataset_name + '/' + video_name, scale_factor=opt.scale_factor)
            test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
            psnr, ssim = demo_test(net, test_loader, opt.scale_factor, dataset_name, video_name)
            PSNR_dataset.append(psnr)
            SSIM_dataset.append(ssim)
        print(dataset_name + ' psnr: ' + str(float(np.array(PSNR_dataset).mean())) + '  ssim: ' + str(float(np.array(SSIM_dataset).mean())))

    if dataset_name == 'Vimeo':
        with open(opt.test_dataset_dir_Vimeo + '/' + dataset_name + '/sep_testlist.txt', 'r') as f:
            video_list = f.read().splitlines()
        for i in range(len(video_list)):
            video_name = video_list[i]
            test_set = TestSetLoader_Vimeo(opt.test_dataset_dir_Vimeo + '/' + dataset_name, video_name, scale_factor=opt.scale_factor)
            test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
            psnr, ssim = demo_test(net, test_loader, opt.scale_factor, dataset_name, video_name)
            PSNR_dataset.append(psnr)
            SSIM_dataset.append(ssim)
        print(dataset_name + ' psnr: ' + str(float(np.array(PSNR_dataset).mean())) + '  ssim: ' + str(float(np.array(SSIM_dataset).mean())))

if __name__ == '__main__':
    for i in range(len(opt.datasets)):
        dataset = opt.datasets[i]
        if not os.path.exists('results/' + dataset):
            os.makedirs('./results/' + dataset)
        import time
        start = time.time()
        main(dataset)
        end = time.time()
        print(end-start)
