import torch
import torchvision.transforms as transforms
from PIL import Image

# 图像加载和预处理函数
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')  # 确保图像为三通道 RGB
    transform = transforms.ToTensor()
    return transform(image)

# 保存图像函数
def save_image(tensor, save_path):
    tensor = tensor.clamp(0, 1)  # 将张量值限制在[0,1]范围
    image = transforms.ToPILImage()(tensor)
    image.save(save_path)

# 等比例缩放较大的图像
def resize_to_smaller(image1, image2):
    # 获取两幅图像的高度和宽度
    h1, w1 = image1.shape[1], image1.shape[2]
    h2, w2 = image2.shape[1], image2.shape[2]
    
    # 确定目标尺寸
    target_h = min(h1, h2)
    target_w = min(w1, w2)
    
    # 定义缩放函数
    resize_transform = transforms.Resize((target_h, target_w))

    # 将较大的图像缩放到较小的图像大小
    if h1 > h2 or w1 > w2:
        image1_resized = resize_transform(transforms.ToPILImage()(image1))
        image1_resized = transforms.ToTensor()(image1_resized)
        image2_resized = image2
    else:
        image2_resized = resize_transform(transforms.ToPILImage()(image2))
        image2_resized = transforms.ToTensor()(image2_resized)
        image1_resized = image1

    return image1_resized, image2_resized

# 计算两幅图像的差值并增强可视化效果
def compute_image_difference(image1_path, image2_path, output_path):
    # 加载图像
    image1 = load_image(image1_path)
    image2 = load_image(image2_path)

    # 等比例缩放较大的图像
    image1_resized, image2_resized = resize_to_smaller(image1, image2)

    # 计算三通道图像差值
    image_diff = torch.abs(image1_resized - image2_resized)

    # 增强差值效果（放大差值图像）
    image_diff = image_diff * 5  # 放大差值图像的值范围，使差异更明显

    # 保存结果图像
    save_image(image_diff, output_path)



# 示例使用
image1_path = './Data/hd/5002.png'  # 图像1路径
image2_path = './Data/duco/5002.jpg'  # 图像2路径
output_path = 'image_diff.jpg'  # 保存的差值图像路径

compute_image_difference(image1_path, image2_path, output_path)
