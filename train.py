import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
import os, argparse
from datetime import datetime

from model.SHLNet_models import SHLNet
# from model.GeleNet_modelsv0 import GeleNet
from data import get_loader
from utils import clip_gradient, adjust_lr
import pytorch_iou
import transforms as trans
from torchinfo import summary
import sys
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
from PIL import Image
import numpy as np
from torch.utils import data
from scipy.ndimage import label as ndi_label
from thop import profile, clever_format
sys.stdout.flush()


# 初始化进程组
def setup(rank, world_size):
    dist.init_process_group(
        backend='nccl',  # NCCL 是 GPU 上的分布式后端
        init_method='env://',  # 使用环境变量初始化
        world_size=world_size,  # 进程总数
        rank=rank  # 当前进程 rank
    )
    torch.cuda.set_device(rank)  # 设置每个进程使用的 GPU

# 清理进程组
def cleanup():
    dist.destroy_process_group()

# 冻结所有前缀为 "sam.image_encoder." 的参数
def freeze_parameters(model, prefix):
    for name, param in model.named_parameters():
        if name.startswith(prefix) and 'adapter' not in name:
            param.requires_grad = False
            # print(f"Froze parameter: {name}")

def fine_tune(rank, world_size, opt):
    setup(rank, world_size)
    model = SHLNet().to(rank)
    load_path = './models/opa/harm/sam_opa_harm.pth.65' 
    # load_path = './models/GeleNet/sam_adp_loss2.pth.107' 
    # load_path = './models/opa/h/sam_opa_h.pth.71' 

    # 加载模型参数到非分布式模型中
    state_dict = torch.load(load_path)
    
    # 如果模型是通过 DistributedDataParallel 保存的，你可能需要去掉前缀 `module.`
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace('module.', '')  # 移除 'module.' 前缀
        new_state_dict[new_key] = state_dict[key]
    
    model.load_state_dict(new_state_dict, strict=True)
    
    freeze_parameters(model, "sam.image_encoder.")

    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    # 优化器，注意这里不需要优化已冻结的参数
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)

    # 加载数据集并使用 DistributedSampler
    train_dataset = get_loader('ISTD/ISTD', './Data/train/', 256, mode='train')
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)  # 分布式采样器
    train_loader = DataLoader(train_dataset, batch_size=opt.batchsize, num_workers=4, pin_memory=True, drop_last=True, sampler=train_sampler)

    # 损失函数
    CE = torch.nn.BCEWithLogitsLoss().to(rank)
    IOU = pytorch_iou.IOU(size_average=True).to(rank)
    #############################################################################################################################
    weight = [1., 1., 0.02, 1.5, 1.5, 0.03]

    #测试数据加载
    # test_dataset = get_loader('test_new/test_DSC', './Data/test/', 256, mode='test')
    # test_loader = data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # 训练循环
    total_step = len(train_loader)
    model.train()
    for epoch in range(1, opt.fine_tune_epoch):
        train_sampler.set_epoch(epoch)  # 每个 epoch 设置不同的采样顺序
        lossp = 0.
        length = 0
        for i, pack in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images, gts, sam_images = pack
            images = images.to(rank)
            gts = gts.to(rank)
            sam_images = sam_images.to(rank)

            # 前向传播
            sal, sal_sig, sam_masks, sam_sig = model(images, sam_images)

            # 计算损失
            loss1 = weight[0] * CE(sal, gts) + weight[1] * IOU(sal_sig, gts) 
            loss2 = weight[3] * CE(sam_masks, gts) + weight[4] * IOU(sam_sig, gts)
            loss = loss1 + loss2

            # 反向传播和优化
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            lossp += loss.item()
            length += 1
            if i % 20 == 0 or i == total_step:
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Learning Rate: {}, Loss1: {:.4f}, Loss2: {:.4f}, Loss: {:.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step,
                           opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch), loss1.item(), loss2.item(), loss.item()))
                           
        print('Epoch [{:03d}/{:03d}], Lossavg:{:.4f}'.format(epoch, opt.epoch, lossp / length))
        # 保存模型
        if (epoch + 1) % 2 == 0 and rank == 0:  # 只有 rank 0 保存模型
            save_path = 'models/ablation/def/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # torch.save(model.state_dict(), save_path + 'sam_casia_img.pth' + '.%d' % epoch)
            torch.save(model.state_dict(), save_path + 'full_model.pth' + '.%d' % epoch)

    if rank == 0:
        cleanup()  # 清理进程组


# 训练函数
def train(rank, world_size, opt):
    setup(rank, world_size)  # 初始化进程组

    # 构建模型并将其移动到当前 GPU 设备
    model = SHLNet().to(rank)
    # 冻结前缀为 "sam.image_encoder." 的参数
    freeze_parameters(model, "sam.image_encoder.")
    # get_model_complexity(model,input_shape=[(1, 3, 256, 256), (1, 3, 512, 512)] )
    device='cuda'
    input_tensor1 = torch.randn((1, 3, 256, 256)).to(device)
    input_tensor2 = torch.randn((1, 3, 512, 512)).to(device)
    summary(model, input_data=(input_tensor1, input_tensor2))  # 根据你的输入数据维度修改
    return 0

    # 使用 DistributedDataParallel，启用 find_unused_parameters=True
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    # 优化器，注意这里不需要优化已冻结的参数
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)

    # 加载数据集并使用 DistributedSampler
    train_dataset = get_loader('ISTD/ISTD', './Data/train/', 256, mode='train')
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)  # 分布式采样器
    train_loader = DataLoader(train_dataset, batch_size=opt.batchsize, num_workers=4, pin_memory=True, drop_last=True, sampler=train_sampler)

    # 损失函数
    CE = torch.nn.BCEWithLogitsLoss().to(rank)
    IOU = pytorch_iou.IOU(size_average=True).to(rank)
    ##########################################################################################################################################
    weight = [1., 1., 0.02, 1.5, 1.5, 0.03]
    # weight = [1., 1., 0.02, 1.0, 1.0, 0.02]
    # weight = [1., 1., 0.02, 0.5, 0.5, 0.02]
    wIOU = 1.
    #测试数据加载
    test_dataset = get_loader('test_new/test_DSC', './Data/test/', 256, mode='test')
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # 训练循环
    total_step = len(train_loader)
    for epoch in range(1, opt.epoch):
        model.train()
        train_sampler.set_epoch(epoch)  # 每个 epoch 设置不同的采样顺序
        lossp = 0.
        length = 0
        for i, pack in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images, gts, sam_images = pack
            images = images.to(rank)
            gts = gts.to(rank)
            sam_images = sam_images.to(rank)

            # 前向传播
            sal, sal_sig, sam_masks, sam_sig = model(images, sam_images)

            # 计算损失
            loss1 = weight[0] * CE(sal, gts) + weight[1] * wIOU * IOU(sal_sig, gts) 
            loss2 = weight[3] * CE(sam_masks, gts) + weight[4] * wIOU * IOU(sam_sig, gts)
            loss = loss1 + loss2

            # 反向传播和优化
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            lossp += loss.item()
            length += 1
            if i % 20 == 0 or i == total_step:
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Learning Rate: {}, Loss1: {:.4f}, Loss2: {:.4f}, Loss: {:.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step,
                           opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch), loss1.item(), loss2.item(), loss.item()))
                           
        print('Epoch [{:03d}/{:03d}], Lossavg:{:.4f}'.format(epoch, opt.epoch, lossp / length))
        # 保存模型##################################################################################################################################
        if (epoch + 1) % 5 == 0 and rank == 0:  # 只有 rank 0 保存模型
            save_path = 'models/opa/lambdaIOU15/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), save_path + 'sam_lambdaIOU15.pth' + '.%d' % epoch)

    if rank == 0:
        cleanup()  # 清理进程组

# 主函数：使用多进程启动训练
def main():
    os.environ["MASTER_ADDR"] = 'localhost'  # 设置主节点地址
    os.environ["MASTER_PORT"] = '12356'  # 设置通信端口，可以选择一个未被占用的端口
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 仅使用 GPU 2 和 3

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50, help='epoch number')
    parser.add_argument('--fine_tune_epoch', type=int, default=60, help='fine_tune epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=3809, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.2, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.2, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=60, help='every n epochs decay learning rate')
    opt = parser.parse_args()

    world_size = torch.cuda.device_count()  # 获取可用的 GPU 数量
    # train(0, 2, opt)
    mp.spawn(train, args=(world_size, opt), nprocs=world_size, join=True)  # 使用 mp.spawn 启动多进程noh
    # mp.spawn(fine_tune, args=(world_size, opt), nprocs=world_size, join=True)  # 使用 mp.spawn 启动多进程noh
    

if __name__ == "__main__":
    main()

def get_model_complexity(model, input_shape, device='cuda'):

    """

    计算模型的参数量和 FLOPs

    

    Args:

        model: PyTorch 模型实例

        input_shape: 输入形状 tuple, 例如 (1, 3, 224, 224) 对应 (batch, channel, height, width)

        device: 运行设备 'cuda' 或 'cpu'

    

    Returns:

        params_str: 格式化后的参数量字符串 (e.g., "25.6 M")

        flops_str: 格式化后的 FLOPs 字符串 (e.g., "4.2 G")

    """

    # 1. 计算参数量 (Parameters)

    # sum(p.numel() for p in model.parameters()) 计算所有参数

    # sum(p.numel() for p in model.parameters() if p.requires_grad) 仅计算可训练参数

    total_params = sum(p.numel() for p in model.parameters())

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    

    # 2. 计算 FLOPs (使用 thop)

    # 创建随机输入张量

    input_tensor1 = torch.randn(*input_shape[0]).to(device)
    input_tensor2 = torch.randn(*input_shape[1]).to(device)
    model.to(device)

    model.eval() # 设置为评估模式，防止 Dropout 等层影响统计

    

    # profile 返回 macs (Multiply-Accumulates) 和 params

    # 注意：FLOPs ≈ 2 * MACs (一次乘加运算包含一次乘法和一次加法)

    # thop 的 profile 函数默认返回的是 MACs，但有时人们口语混用。

    # clever_format 会自动格式化数字

    macs, params = profile(model, inputs=(input_tensor1, input_tensor2 ), verbose=False)

    

    # 格式化输出

    flops_str, params_str = clever_format([macs, params])

    

    # 如果需要严格的 FLOPs (而不是 MACs)，通常乘以 2

    # flops_val = macs * 2 

    # 但学术界汇报常直接汇报 MACs 或标注为 FLOPs (视具体领域习惯而定)

    # 这里我们直接返回 thop 计算的原始值 (通常是 MACs)
    print(total_params, flops_str)
    

    return {

        "total_params": total_params,

        "trainable_params": trainable_params,

        "params_formatted": params_str,

        "macs_formatted": flops_str, # 这里实际上是 MACs

        "macs_raw": macs

    }