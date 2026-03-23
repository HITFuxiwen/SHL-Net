import torch
import torch.nn.functional as F
from torch.utils import data
import os, argparse
from torchvision import transforms

# from model.SHLNet_modelsv0 import SHLNet
from model.SHLNet_models import SHLNet
from data import get_loader
import transforms as trans
import imageio
from main import evaluate
from evalnew import calculate_auc_iou_fpr
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
import numpy as np

def preprocess_gt(gt_path):
    # Read GT image and normalize pixel values to 0 or 1
    gt_img_normalized = Image.open(gt_path).convert('L')
    # gt_img_normalized = np.array(gt_img) / 255.0
    return gt_img_normalized

def preprocess_pred(pred_path):
    # Read pred image and normalize pixel values to 0-1 range
    # pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    pred_img_normalized = Image.open(pred_path).convert('L')
    # pred_img_normalized = np.array(pred_img) / 255.0
    return pred_img_normalized

def calculate_iou(gt_flat, pred_binary):
    intersection = np.logical_and(gt_flat, pred_binary).sum()
    union = np.logical_or(gt_flat, pred_binary).sum()
    iou = intersection / (union + 1e-8)  # Add a small epsilon to avoid division by zero
    return iou

def calculate_fpr(gt_flat, pred_binary):
    true_negative = np.logical_and(np.logical_not(gt_flat), np.logical_not(pred_binary)).sum()
    false_positive = np.logical_and(np.logical_not(gt_flat), pred_binary).sum()
    fpr = false_positive / (false_positive + true_negative + 1e-8)  # Add a small epsilon to avoid division by zero
    return fpr


# 设置可见的 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=1, help='testing size')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
opt = parser.parse_args()


# 初始化模型（不使用分布式）
model = SHLNet()

for epoch in range(9, 80, 5):
    # load_path = './models/casia/img/imgsam_casia_img.pth.' + str(epoch)
    load_path = './models/opa/lambdaIOU15/sam_lambdaIOU15.pth.' + str(epoch)
    # load_path = './models/ablation/opa/opasam_LMSA_fuse.pth.' + str(epoch)

    # 加载模型参数到非分布式模型中
    state_dict = torch.load(load_path)
    
    # 如果模型是通过 DistributedDataParallel 保存的，你可能需要去掉前缀 `module.`
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace('module.', '')  # 移除 'module.' 前缀
        new_state_dict[new_key] = state_dict[key]
    
    model.load_state_dict(new_state_dict, strict=True)

    # 将模型移动到 GPU 并设置为评估模式
    model.cuda()
    model.eval()

    # 加载测试数据集
    test_dataset = get_loader('', './Data/test/', 256, mode='test')
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)

    print('''
            Starting testing:
                dataset: {}
                Testing size: {}
            '''.format('ISTD/test', len(test_loader.dataset)))

    # 设置保存路径
    save_path = './res/'
    # save_path = './xuantu2/wottd/def/'
    save_path2 = './test/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_path2):
        os.makedirs(save_path2)

    # 测试循环
    with torch.no_grad():  # 禁用梯度计算
        auc_scores = []
        # auc_scores = {'t_image': [], 'test_D': [], 'test_h': []}  # AUC scores for each pred image
        f1_scores = []  # F1 scores for each pred image
        mcc_scores = []  # MCC scores for each pred image
        iou_scores = []  # IoU scores for each pred image
        fpr_scores = []  # FPR scores for each pred image
        # auc_scores = {'t_image': [], 'test_D': [], 'test_h': [], 'def_image':[], 'Casia_tp': [], 'test_casia_img': [], 'test_def_img': []} 
        # f1_scores = {'t_image': [], 'test_D': [], 'test_h': [], 'def_image':[], 'Casia_tp': [], 'test_casia_img': []} 
        # mcc_scores = {'t_image': [], 'test_D': [], 'test_h': [], 'def_image':[], 'Casia_tp': [], 'test_casia_img': []} 
        # iou_scores = {'t_image': [], 'test_D': [], 'test_h': [], 'def_image':[], 'Casia_tp': [], 'test_casia_img': []} 
        # fpr_scores = {'t_image': [], 'test_D': [], 'test_h': [], 'def_image':[], 'Casia_tp': [], 'test_casia_img': []} 
        for i, data_batch in enumerate(test_loader):
            image, sam_images, image_w, image_h, image_path, gt = data_batch

            # 打印输入的图像大小，方便调试
            # print(image.shape)

            # 获取保存文件路径
            savedir = image_path[0].split('/')[-2]
            filename = image_path[0].split('/')[-1].split('.')[0]

            # 将数据移动到 GPU
            image = image.cuda()
            sam_images = sam_images.cuda()

            # 前向传播
            res, sal_sig, sam_masks, sam_sig = model(image, sam_images)

            # 后处理并将结果保存到 CPU
            res = sal_sig.data.cpu().squeeze(0)
            sam_sig = sam_sig[0].detach().cpu()
            # 图像保存大小
            image_w, image_h = 256, 256
            transform = trans.Compose([
                transforms.ToPILImage(),
                trans.Scale((image_w, image_h))
            ])

            # 将结果转换为 PIL 图像并保存
            res = transform(res).convert('RGB')
            sam_sig2 = transform(sam_sig).convert('RGB')

            save2 = save_path2 + savedir + '/'
            save1 = save_path + savedir + '/'
            if not os.path.exists(save2):
                os.makedirs(save2)
            if not os.path.exists(save1):
                os.makedirs(save1)

            res.save(save2 + filename + '.jpg')
            sam_sig2.save(save1 + 'sam_' + filename + '.jpg')

            gt_flat = (gt.cpu().numpy().flatten() > 0.5).astype(int)
            pred_flat = sam_sig.numpy().flatten()
            gt_flat = gt_flat.astype(int)

            # Calculate AUC for the current pred image
            if len(np.unique(gt_flat)) < 2:
                auc_score = 0.5
            else:
                auc_score = roc_auc_score(gt_flat, pred_flat)
            if auc_score < 0.5:
                auc_score = 1 - auc_score
            auc_scores.append(auc_score)

            # Calculate F1 score for the current pred image
            threshold = 0.5  # Set the threshold for binarization
            pred_binary = (pred_flat > threshold).astype(int)
            f1 = f1_score(gt_flat, pred_binary)
            f1_scores.append(f1)

            # Calculate MCC for the current pred image
            mcc = matthews_corrcoef(gt_flat, pred_binary)
            mcc_scores.append(mcc)

            # Calculate IoU for the current pred image
            iou = calculate_iou(gt_flat, pred_binary)
            iou_scores.append(iou)

            # Calculate FPR for the current pred image
            fpr = calculate_fpr(gt_flat, pred_binary)
            fpr_scores.append(fpr)

                    # print(f"Image {gt_files[i]} - AUC: {auc_score:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}, IoU: {iou:.4f}, FPR: {fpr:.4f}")

                # Calculate the average AUC, F1, MCC, IoU, and FPR
        average_auc = np.mean(auc_scores)
        average_f1 = np.mean(f1_scores)
        average_mcc = np.mean(mcc_scores)
        average_iou = np.mean(iou_scores)
        average_fpr = np.mean(fpr_scores)
        print("Average AUC:", average_auc)
        print("Average F1:", average_f1)
        print("Average MCC:", average_mcc)
        print("Average IoU:", average_iou)
        print("Average FPR:", average_fpr)

        print('Testing completed for epoch {}'.format(epoch))

