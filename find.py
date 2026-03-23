import os
import numpy as np
from sklearn.metrics import f1_score
from collections import defaultdict
from skimage.io import imread
from skimage.transform import resize  # 导入resize功能
from skimage.color import rgb2gray
from PIL import Image
import numpy as np
# 设置目录路径
# ground_truth_folder = 'Data/test/t_gt'  # Ground Truth 文件夹路径
# my_algorithm_folder = 'xuantu2/full/opa/t_image'  # 我的算法文件夹路径
# other_algorithms_folders = [
#     'xuantu2/wlmsa/opa/t_image' ,  # 其他算法1
#     'xuantu2/wofasa/opa/t_image' ,  # 其他算法2
#     'xuantu2/womvff/opa/t_image' 
# ]

# ground_truth_folder = 'Data/test/test_casia_gt'  # Ground Truth 文件夹路径
# my_algorithm_folder = 'xuantu2/full/casia/test_casia_img'  # 我的算法文件夹路径
# other_algorithms_folders = [
#     'xuantu2/wlmsa/casia/test_casia_img' ,  # 其他算法1
#     'xuantu2/wofasa/casia/test_casia_img' ,  # 其他算法2
#     'xuantu2/womvff/casia/test_casia_img' 
# ]

ground_truth_folder = 'Data/test/test_def_gt'  # Ground Truth 文件夹路径
my_algorithm_folder = 'xuantu2/full/def/test_def_img'  # 我的算法文件夹路径
other_algorithms_folders = [
    'xuantu2/wlmsa/def/test_def_img' ,  # 其他算法1
    'xuantu2/wofasa/def/test_def_img' ,  # 其他算法2
    'xuantu2/womvff/def/test_def_img' 
]
# 获取文件列表
def get_file_list(folder_path):
    return [f for f in os.listdir(folder_path) if f.endswith('.png') or f.endswith('.jpg')]  # 假设文件是图像格式

# 计算F1分数
def calculate_f1(pred_mask, gt_mask):
    # 确保输入是二值图像
    pred_mask = (pred_mask > 0).astype(int)  # 将大于0的值设置为1，其他为0
    gt_mask = (gt_mask > 0).astype(int)  # 将大于0的值设置为1，其他为0
    return f1_score(gt_mask.flatten(), pred_mask.flatten(), average='binary')

# 加载并调整图像尺寸
def load_and_resize_image(file_path, target_size=(256, 256)):
    # 使用PIL打开图像并转换为灰度图像
    img = Image.open(file_path).convert('L')  # 'L' 模式表示灰度图
    
    # 调整图像大小
    img_resized = img.resize(target_size)  # resize为(256, 256)
    
    # 转换为NumPy数组
    img_array = np.array(img_resized)
    
    return img_array

# 获取每个算法的F1分数
def get_f1_scores(ground_truth_folder, my_algorithm_folder, other_algorithms_folders):
    gt_files = get_file_list(ground_truth_folder)
    my_f1_scores = {}
    other_f1_scores = {folder: [] for folder in other_algorithms_folders}
    
    # 优化：提前计算每个文件的F1分数，避免重复计算
    gt_masks = {gt_file: load_and_resize_image(os.path.join(ground_truth_folder,  gt_file)) for gt_file in gt_files}
    my_masks = {gt_file: load_and_resize_image(os.path.join(my_algorithm_folder, 'sam_' + gt_file)) for gt_file in gt_files}
    
    # 计算我的算法和其他算法的F1分数
    for gt_file in gt_files:
        gt_mask = gt_masks[gt_file]
        my_mask = my_masks[gt_file]
        # 计算我的算法的F1分数
        my_f1_scores[gt_file] = calculate_f1(my_mask, gt_mask)
        
        # 计算其他算法的F1分数
        for folder in other_algorithms_folders:
            algorithm_mask = load_and_resize_image(os.path.join(folder, 'sam_' + gt_file))
            f1 = calculate_f1(algorithm_mask, gt_mask)
            other_f1_scores[folder].append((gt_file, f1))
    
    return my_f1_scores, other_f1_scores

# 比较F1分数，输出差距最大的10个图片
def compare_f1_and_find_top10(my_f1_scores, other_f1_scores):
    differences = []

    # 计算每个文件的差距，并计算最大差距
    for gt_file, my_f1 in my_f1_scores.items():
        max_diff = -1e6
        max_diff_algorithm = ''
        
        # 向量化计算F1差距
        all_diffs = []
        for algorithm, results in other_f1_scores.items():
            # 获取该算法的F1分数
            f1_scores = {result_file: f1 for result_file, f1 in results}
            f1 = f1_scores.get(gt_file, 0)
            diff = my_f1 - f1
            all_diffs.append(diff)
        
        # 获取最大差距的算法
        max_diff = min(all_diffs)
        differences.append((gt_file, max_diff, my_f1))
    
    # 按照差距排序，取最大的10个
    differences.sort(key=lambda x: x[1], reverse=True)
    
    # 输出前10个差距最大的图片名称
    top10_files = [diff[0] for diff in differences[:50]]
    return top10_files

# 主程序
def main():
    # 获取所有F1分数
    my_f1_scores, other_f1_scores = get_f1_scores(ground_truth_folder, my_algorithm_folder, other_algorithms_folders)
    
    # 找到差距最大的10个图片
    top10_files = compare_f1_and_find_top10(my_f1_scores, other_f1_scores)
    
    # 输出结果
    print("The top 10 images with the largest F1 score differences are:")
    for img in top10_files:
        print(img)

# 执行主程序
if __name__ == "__main__":
    main()
