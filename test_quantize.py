from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import *

from model_quantize import *
from evaluation import psnr2, ssim
import numpy as np
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import argparse, os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import quantize as quant_iao
parser = argparse.ArgumentParser(description="PyTorch D3Dnet")
parser.add_argument("--scale_factor", type=int, default=4, help="scale")
parser.add_argument("--valid_dataset_dir", default='/Datadisk/zhangsiyu/datasets/Vimeo', type=str, help="valid_dataset")
parser.add_argument("--test_dataset_dir", default='./data', type=str, help="test_dataset dir")
parser.add_argument("--test_dataset_dir_Vimeo", default='/Datadisk/zhangsiyu/datasets', type=str, help="test_dataset dir")
parser.add_argument("--model", default='./log/D3Dnet.pth.tar', type=str, help="checkpoint")
parser.add_argument("--inType", type=str, default='y', help="RGB input or y input")
parser.add_argument("--batchSize", type=int, default=8, help="Test batch size")
parser.add_argument("--gpu", type=int, default=7, help="Test batch size")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
# parser.add_argument("--datasets", type=str, default=['Vid4','SPMC-11','Vimeo'], help="Test batch size")
parser.add_argument("--datasets", type=str, default=['Vid4'], help="Test batch size")

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
parser.add_argument("--ptq_batch", type=int, default=50, help="the batch of ptq")

global opt, model
opt = parser.parse_args()
torch.cuda.set_device(opt.gpu)
# USE_CUDA = torch.cuda.is_available()
# device = torch.device("cuda:0" if USE_CUDA else "cpu")

def demo_test(net, test_loader, scale_factor, dataset_name, video_name):
    PSNR_list = []
    SSIM_list = []
    with torch.no_grad():
        for idx_iter, (LR, HR, SR_buicbic) in enumerate(test_loader):
            # print(idx_iter)
            if idx_iter > 0:
                break
            LR, HR = Variable(LR).cuda(), Variable(HR).cuda()
            # LR, HR = Variable(LR).to(device), Variable(HR).to(device)
            SR = net(LR)   #LR->SR，同时SR是单通道，应该是Y通道
            SR = torch.clamp(SR, 0, 1) #0和1的截断函数

            PSNR_list.append(psnr2(SR, HR[:, :, 3, :, :])) #评价指标仅仅在Y通道
            SSIM_list.append(ssim(SR, HR[:, :, 3, :, :]))
            print(1)

            # if not os.path.exists('results/' + dataset + '/' + video_name):
            #     os.makedirs('./results/' + dataset + '/' + video_name)
            ## save y images
            # SR_img = transforms.ToPILImage()(SR[0, :, :, :].cpu())
            # SR_img.save('results/' + dataset_name + '/' + video_name + '/sr_y_' + str(idx_iter+1).rjust(2, '0') + '.png')
            #
            # ## save rgb images
            # SR_buicbic[:, 0, :, :] = SR[:, 0, :, :].cpu()   #用计算的Y通道的SR结果替换对LR插值的结果
            # SR_rgb = (ycbcr2rgb(SR_buicbic[0,:,:,:].permute(2,1,0))).permute(2,1,0)
            # SR_rgb = torch.clamp(SR_rgb, 0, 1)
            # SR_img = transforms.ToPILImage()(SR_rgb)
            # SR_img.save('results/' + dataset_name + '/' + video_name + '/sr_rgb_' + str(idx_iter+1).rjust(2, '0') + '.png')

        PSNR_mean = float(torch.cat(PSNR_list, 0)[2:-2].data.cpu().mean())
        SSIM_mean = float(torch.cat(SSIM_list, 0)[2:-2].data.cpu().mean())
        print(video_name + ' psnr: '+ str(PSNR_mean) + ' ssim: ' + str(SSIM_mean))
        return PSNR_mean, SSIM_mean

def ptq(net):
    valid_set = ValidSetLoader(opt.ptq_dataset_dir, scale_factor=opt.scale_factor, inType=opt.inType)
    valid_loader = DataLoader(dataset=valid_set, num_workers=opt.threads, batch_size=10, shuffle=True)
    PSNR_list = []
    SSIM_list = []
    for idx_iter, (LR, HR) in enumerate(valid_loader):
        LR, HR = Variable(LR).cuda(), Variable(HR).cuda()
        SR = net(LR)
        SR = torch.clamp(SR, 0, 1)  # 0和1的截断函数

        PSNR_list.append(psnr2(SR, HR[:, :, 3, :, :]))  # 评价指标仅仅在Y通道
        SSIM_list.append(ssim(SR, HR[:, :, 3, :, :]))
        # if idx_iter > opt.ptq_batch:
        #     break
        print("Batch:", idx_iter)
    # print("1")

def main(dataset_name):
    net1 = Net(opt.scale_factor)
    # net1 = torch.nn.DataParallel(net1, device_ids=[4, 5, 6, 7])
    net = Net(opt.scale_factor)
    # net = torch.nn.DataParallel(net, device_ids=[4, 5, 6, 7])
    model = torch.load(opt.model)
    net1.load_state_dict(model['state_dict'])
    net.load_state_dict(model['state_dict'])
    print("***ori_model***\n", net1)
    quant_iao.prepare(
        net,
        inplace=True,
        a_bits=16,
        w_bits=16,
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

    # epoch_state = 0
    net1 = net1.cuda()
    net = net.cuda()
    # net1 = torch.nn.DataParallel(
    #     net1, device_ids=range(torch.cuda.device_count())
    # )
    print("ptq is doing...")
    # ptq(net1)
    # valid_set = ValidSetLoader(opt.valid_dataset_dir, scale_factor=opt.scale_factor, inType=opt.inType)
    # valid_loader = DataLoader(dataset=valid_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)
    # PSNR_list1 = []
    # # SSIM_list = []
    # PSNR_list2 = []
    #
    # for idx_iter, (LR, HR) in enumerate(valid_loader):
    #     LR, HR = Variable(LR).cuda(), Variable(HR).cuda()
    #     SR = net(LR)
    #     SR = torch.clamp(SR, 0, 1)  # 0和1的截断函数
    #     SR1 = net1(LR)
    #     PSNR_list1.append(psnr2(SR.detach(), HR[:, :, 3, :, :].detach()))  # 评价指标仅仅在Y通道
    #     PSNR_list2.append(psnr2(SR1.detach(), HR[:, :, 3, :, :].detach()))
    #     # SSIM_list.append(ssim(SR, HR[:, :, 3, :, :])) #SSIM要慎用，会爆显存
    #     torch.cuda.empty_cache()
    #     if idx_iter > opt.ptq_batch:
    #         break
    #     print("Batch:", idx_iter)
    #
    # # aa = net.state_dict()
    # # bb = model['state_dict']
    # print('ori PSNR---%f' % (torch.mean(torch.stack(PSNR_list2))))
    # print('ptq PSNR---%f' % (torch.mean(torch.stack(PSNR_list1))))


        # psnr_list.append(psnr(SR.detach(), HR[:, :, 3, :, :].detach()))
    # print('valid PSNR---%f' % (float(np.array(psnr_list).mean())))
    PSNR_dataset = []
    SSIM_dataset = []
    if dataset_name == 'Vid4' or dataset_name == 'SPMC-11':
        video_list = os.listdir(opt.test_dataset_dir + '/' + dataset_name)
        # for i in range(0, len(video_list)):
        for i in range(0, 1):
            print(i)
            video_name = video_list[i]
            test_set = TestSetLoader(opt.test_dataset_dir + '/' + dataset_name + '/' + video_name, scale_factor=opt.scale_factor)
            test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
            psnr, ssim = demo_test(net1, test_loader, opt.scale_factor, dataset_name, video_name)
            PSNR_dataset.append(psnr)
            SSIM_dataset.append(ssim)
        print(dataset_name + ' psnr: ' + str(float(np.array(PSNR_dataset).mean())) + '  ssim: ' + str(float(np.array(SSIM_dataset).mean())))
    #
    # PSNR_dataset = []
    # SSIM_dataset = []
    # if dataset_name == 'Vimeo':
    #     with open(opt.test_dataset_dir_Vimeo + '/' + dataset_name + '/sep_testlist.txt', 'r') as f:
    #         video_list = f.read().splitlines()
    #     for i in range(len(video_list)):
    #         video_name = video_list[i]
    #         test_set = TestSetLoader_Vimeo(opt.test_dataset_dir_Vimeo + '/' + dataset_name, video_name, scale_factor=opt.scale_factor)
    #         test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=True)
    #         psnr, ssim = demo_test(net, test_loader, opt.scale_factor, dataset_name, video_name)
    #         PSNR_dataset.append(psnr)
    #         SSIM_dataset.append(ssim)
    #     print(dataset_name + ' psnr: ' + str(float(np.array(PSNR_dataset).mean())) + '  ssim: ' + str(float(np.array(SSIM_dataset).mean())))

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
