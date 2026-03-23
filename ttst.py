import torch
import numpy as np
import scipy.ndimage as ndi
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
# 定义连通性损失函数
def connectivity_loss(pred_mask, dilation_size=3):
    """
    计算掩模的连通组件数量
    :param pred_mask: 二值化掩模张量
    :param dilation_size: 膨胀的大小，调整以连接短距离间隔的目标区域
    :return: 连通性损失
    """
    # 先将掩模转为二值图像
    pred_mask_bin = pred_mask > 0.5
    
    # 对掩模进行膨胀处理，填补小间隙
    # pred_mask_dilated = F.conv2d(pred_mask_bin.unsqueeze(0).unsqueeze(0).float(), 
    #                           weight=torch.ones(1, 1, dilation_size, dilation_size).float(), 
    #                           padding=dilation_size//2).squeeze(0).squeeze(0) > 0
    
    pred_mask_dilated = pred_mask_bin

    # 保存膨胀后的图像
    save_image(pred_mask_dilated.cpu().numpy(), 'dilated_mask.jpg')
    # 计算连通区域的数量
    labeled, num_features = ndi.label(pred_mask_dilated.cpu().numpy())
    # labeled_image = Image.fromarray((labeled * (255)).astype(np.uint8))
    # labeled_image.save("labeled.png")
    # 如果目标区域超过1个，则增加惩罚
    loss = num_features - 1
    return max(0, loss)

def SoftCL(pred_mask, kernel_size=3):
    """
    计算软连通性损失，鼓励输出的掩模图像形成较少的连通块
    :param pred_mask: [B, 1, H, W] 的概率掩模张量 (取值范围 0~1)
    :param kernel_size: 计算局部相似性的窗口大小
    :return: 软连通性损失（可微分）
    """
    # 使用均值卷积来计算局部一致性
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=pred_mask.device) / (kernel_size ** 2)
    
    # 计算局部平均值
    smooth_mask = F.conv2d(pred_mask, kernel, padding=kernel_size // 2)

    # 计算局部连通性：邻域一致性越高，表示连通性越强
    diff = torch.abs(pred_mask - smooth_mask)

    # 计算软连通性损失：鼓励邻域差异更小
    connectivity_loss = diff.mean()

    return connectivity_loss

import torch
import torch.nn.functional as F

def SoftCL2(pred_mask, kernel_size=3, eps=1e-6, alpha=10):
    """
    计算归一化的软连通性损失，并优化小目标的损失权重
    :param pred_mask: [B, H, W] 或 [B, 1, H, W] 的概率掩模张量 (取值范围 0~1)
    :param kernel_size: 计算局部相似性的窗口大小
    :param eps: 防止除零错误
    :param alpha: 避免小目标损失过大的平衡因子
    :return: 归一化的软连通性损失
    """
    if pred_mask.dim() == 3:
        pred_mask = pred_mask.unsqueeze(1)

    # 平滑掩模，估计局部连通性
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=pred_mask.device) / (kernel_size ** 2)
    smooth_mask = F.conv2d(pred_mask, kernel, padding=kernel_size // 2)

    # 计算局部连通性损失
    diff = torch.abs(pred_mask - smooth_mask)

    # 计算目标区域大小（mask_size 归一化，避免过小影响过大）
    mask_size = torch.sum(pred_mask > 0.1, dim=(1, 2, 3)) + eps

    # 归一化损失
    connectivity_loss = torch.sum(diff, dim=(1, 2, 3)) / (mask_size + alpha) ** (1/4)
    return connectivity_loss.mean()

def compute_bbox(mask):
    """
    计算每个 mask 的边界框 (bounding box)，用于计算宽高比
    :param mask: [B, 1, H, W] 归一化的 mask
    :return: width, height
    """
    B, _, H, W = mask.shape
    width_list, height_list = [], []
    
    for b in range(B):  # 遍历 batch
        binary_mask = (mask[b, 0] > 0.1).float()  # 阈值化处理
        
        # 计算高度范围
        y_indices = torch.any(binary_mask, dim=1).nonzero(as_tuple=True)[0]  # 找到包含前景的行索引
        if len(y_indices) > 0:
            ymin, ymax = y_indices[0], y_indices[-1]
            height = ymax - ymin + 1
        else:
            height = 1  # 避免除零错误
        
        # 计算宽度范围
        x_indices = torch.any(binary_mask, dim=0).nonzero(as_tuple=True)[0]  # 找到包含前景的列索引
        if len(x_indices) > 0:
            xmin, xmax = x_indices[0], x_indices[-1]
            width = xmax - xmin + 1
        else:
            width = 1

        width_list.append(width)
        height_list.append(height)
    
    width_tensor = torch.tensor(width_list, dtype=torch.float, device=mask.device)
    height_tensor = torch.tensor(height_list, dtype=torch.float, device=mask.device)
    
    return width_tensor, height_tensor

def SoftCL_Thinness(pred_mask, kernel_size=3, eps=1e-6, alpha=10, beta=0.5):
    """
    计算软连通性损失，并降低对细长形状的惩罚
    :param pred_mask: [B, 1, H, W] 概率掩模
    :param kernel_size: 平滑窗口大小
    :param eps: 防止除零错误
    :param alpha: 避免小目标损失过大的平衡因子
    :param beta: 细长形状的损失衰减因子
    :return: 软连通性损失
    """
    if pred_mask.dim() == 3:
        pred_mask = pred_mask.unsqueeze(1)

    # 平滑 mask
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=pred_mask.device) / (kernel_size ** 2)
    smooth_mask = F.conv2d(pred_mask, kernel, padding=kernel_size // 2)

    # 计算局部连通性损失
    diff = torch.abs(pred_mask - smooth_mask)

    # 计算目标区域大小
    mask_size = torch.sum(pred_mask > 0.1, dim=(1, 2, 3)) + eps

    # 计算真实目标的宽高
    width, height = compute_bbox(pred_mask)

    # 计算长宽比（细长性）
    thinness = torch.min(width, height) / (torch.max(width, height) + eps)
    # print(thinness)
    # 归一化损失
    connectivity_loss = torch.sum(diff, dim=(1, 2, 3)) / (mask_size + alpha) ** (1/2)
    connectivity_loss = connectivity_loss * (thinness)  # 细长目标降低惩罚
    return connectivity_loss.mean()

# 读取图像并处理
def load_image(image_path):
    # 使用PIL读取图像
    image = Image.open(image_path).convert('L')  # 转为灰度图
    image = np.array(image)
    
    # 假设图像已经是二值图像，值为0和255
    image = image / 255.0  # 转换为[0, 1]之间的浮动
    
    # 转换为Torch tensor
    image_tensor = torch.tensor(image, dtype=torch.float32)
    
    return image_tensor.unsqueeze(0)

def save_image(image_array, filename):
    """
    将图像数组保存为图片
    :param image_array: 图像数组
    :param filename: 保存文件的路径
    """
    image = Image.fromarray((image_array * 255).astype(np.uint8))  # 转为[0, 255]区间的图像
    image.save(filename)
    print(f"膨胀后的图像已保存为 {filename}")
# 测试
for i in range (100):
    image_path = './res/test_D/sam_' + str(i) + '.jpg'  # 替换为你的图片路径
    # image_path = './res/t_image/sam_3.jpg'  # 替换为你的图片路径
    # image_path = './res/sam_4941.jpg'  # 替换为你的图片路径
    image_tensor = load_image(image_path)

    # 计算连通性损失
    loss = SoftCL_Thinness(image_tensor)

    # 输出结果
    print(f'Connectivity loss: {loss}')

# # 可视化图像和连通区域
# plt.imshow(image_tensor.numpy(), cmap='gray')
# plt.title(f'Connectivity Loss: {loss}')
# plt.show()
