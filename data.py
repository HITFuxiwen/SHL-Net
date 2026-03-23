from PIL import Image
from torch.utils import data
import transforms as trans
from torchvision import transforms
import random
import os
import sys
from model.segment_anything.modeling import Sam
import numpy as np
import torch
from typing import Optional, Tuple

from model.segment_anything.utils.transforms import ResizeLongestSide

def load_list(dataset_name, data_root):

    images = []
    labels = []
    


    img_root = data_root + 'images/'
    img_files = os.listdir(img_root)

    for img in img_files:

        images.append(img_root + img)
        labels.append(img_root.replace('/images/', '/groundtruths/') + img)
    
    # img_root = data_root + 'duco/'
    # img_files = os.listdir(img_root)

    # for img in img_files:

    #     images.append(img_root + img)
    #     labels.append(img_root.replace('/duco/', '/groundtruths/') + img)

    # img_root = data_root + 'hd/'
    # img_files = os.listdir(img_root)

    # for img in img_files:

    #     images.append(img_root + img)
    #     labels.append(img_root.replace('/hd/', '/groundtruths/') + img.replace('.png','.jpg'))

#/casia******************************************************************************************************************************************************************/

    # img_root = data_root + 'casia_image/'
    # label_root = data_root + 'casia_gt/'
    # img_files = sorted(os.listdir(img_root))
    # label_files = sorted(os.listdir(label_root))
    # for img in img_files:
    #     images.append(img_root + img)
    # for label in label_files:
    #     labels.append(label_root + label)

    # img_root = data_root + 'casia_D/'
    # label_root = data_root + 'casia_gt/'
    # img_files = sorted(os.listdir(img_root))
    # label_files = sorted(os.listdir(label_root))
    # for img in img_files:
    #     images.append(img_root + img)
    # for label in label_files:
    #     labels.append(label_root + label)
    
    # img_root = data_root + 'casia_h/'
    # label_root = data_root + 'casia_gt/'
    # img_files = sorted(os.listdir(img_root))
    # label_files = sorted(os.listdir(label_root))
    # for img in img_files:
    #     images.append(img_root + img)
    # for label in label_files:
    #     labels.append(label_root + label)

#/defacto*************************************************************************************************************************************************************/
    # img_root = data_root + 'def_image/'
    # label_root = data_root + 'def_gt/'
    # img_files = sorted(os.listdir(img_root))
    # label_files = sorted(os.listdir(label_root))
    # for img in img_files:
    #     images.append(img_root + img)
    # for label in label_files:
    #     labels.append(label_root + label)

    # img_root = data_root + 'def_D/'
    # label_root = data_root + 'def_gt/'
    # img_files = sorted(os.listdir(img_root))
    # label_files = sorted(os.listdir(label_root))
    # for img in img_files:
    #     images.append(img_root + img)
    # for label in label_files:
    #     labels.append(label_root + label)
    
    # img_root = data_root + 'def_h/'
    # label_root = data_root + 'def_gt/'
    # img_files = sorted(os.listdir(img_root))
    # label_files = sorted(os.listdir(label_root))
    # for img in img_files:
    #     images.append(img_root + img)
    # for label in label_files:
    #     labels.append(label_root + label)

    return images, labels

def load_test_list(test_path, data_root):

    images = []
    labels = []
    
    
    img_root = data_root + 't_image/'
    img_files = os.listdir(img_root)

    for img in img_files:

        images.append(img_root + img)
        labels.append(img_root.replace('/t_image/', '/t_gt/') + img)
    
    
    # img_root = data_root + 'test_D/'
    # img_files = os.listdir(img_root)

    # for img in img_files:

    #     images.append(img_root + img)
    #     labels.append(img_root.replace('/test_D/', '/t_gt/') + img)
    
    # img_root = data_root + 'test_h/'
    # img_files = os.listdir(img_root)

    # for img in img_files:

    #     images.append(img_root + img)
    #     labels.append(img_root.replace('/test_h/', '/t_gt/') + img.replace('.png','.jpg'))

    #/casia******************************************************************************************************************************************************************/

    # img_root = data_root + 'test_casia_img/'
    # label_root = data_root + 'test_casia_gt/'
    # img_files = sorted(os.listdir(img_root))
    # label_files = sorted(os.listdir(label_root))
    # for img in img_files:
    #     images.append(img_root + img)
    # for label in label_files:
    #     labels.append(label_root + label)

    # img_root = data_root + 'test_casia_D/'
    # label_root = data_root + 'test_casia_gt/'
    # img_files = sorted(os.listdir(img_root))
    # label_files = sorted(os.listdir(label_root))
    # for img in img_files:
    #     images.append(img_root + img)
    # for label in label_files:
    #     labels.append(label_root + label)
    
    # img_root = data_root + 'test_casia_h/'
    # label_root = data_root + 'test_casia_gt/'
    # img_files = sorted(os.listdir(img_root))
    # label_files = sorted(os.listdir(label_root))
    # for img in img_files:
    #     images.append(img_root + img)
    # for label in label_files:
    #     labels.append(label_root + label)

#/defacto*************************************************************************************************************************************************************/
    # img_root = data_root + 'test_def_img/'
    # label_root = data_root + 'test_def_gt/'
    # img_files = sorted(os.listdir(img_root))
    # label_files = sorted(os.listdir(label_root))
    # for img in img_files:
    #     images.append(img_root + img)
    # for label in label_files:
    #     labels.append(label_root + label)

    # img_root = data_root + 'test_def_D/'
    # label_root = data_root + 'test_def_gt/'
    # img_files = sorted(os.listdir(img_root))
    # label_files = sorted(os.listdir(label_root))
    # for img in img_files:
    #     images.append(img_root + img)
    # for label in label_files:
    #     labels.append(label_root + label)
    
    # img_root = data_root + 'test_def_h/'
    # label_root = data_root + 'test_def_gt/'
    # img_files = sorted(os.listdir(img_root))
    # label_files = sorted(os.listdir(label_root))
    # for img in img_files:
    #     images.append(img_root + img)
    # for label in label_files:
    #     labels.append(label_root + label)

    return images, labels
 

class ImageData(data.Dataset):
    def __init__(self, dataset_list, data_root, transform, mode, img_size=None, scale_size=None, t_transform=None):
        if mode == 'train':
            self.image_path, self.label_path = load_list(dataset_list, data_root)
        else:
            self.image_path, self.label_path = load_test_list(dataset_list, data_root)
        self.transform = transform
        self.t_transform = t_transform
        self.sam_transform = ResizeLongestSide(512)
        # self.label_14_transform = label_14_transform
        # self.label_28_transform = label_28_transform
        # self.label_56_transform = label_56_transform
        # self.label_112_transform = label_112_transform
        self.mode = mode
        self.img_size = img_size
        self.scale_size = scale_size

    def __getitem__(self, item):
        fn = self.image_path[item].split('/')

        filename = fn[-1]

        image = Image.open(self.image_path[item]).convert('RGB')
        image_w, image_h = int(image.size[0]), int(image.size[1])
        label = Image.open(self.label_path[item]).convert('L')
        random_size = self.scale_size
        if self.mode == 'train':

            new_img = trans.Scale((random_size, random_size))(image)
            new_label = trans.Scale((random_size, random_size), interpolation=Image.NEAREST)(label)

            # random crop
            # w, h = new_img.size
            # if w != self.img_size and h != self.img_size:
            #     x1 = random.randint(0, w - self.img_size)
            #     y1 = random.randint(0, h - self.img_size)
            #     new_img = new_img.crop((x1, y1, x1 + self.img_size, y1 + self.img_size))
            #     new_label = new_label.crop((x1, y1, x1 + self.img_size, y1 + self.img_size))

            # random flip
            if random.random() < 0.5:
                new_img = new_img.transpose(Image.FLIP_LEFT_RIGHT)
                new_label = new_label.transpose(Image.FLIP_LEFT_RIGHT)
            
            new_img2 = np.asarray(new_img, dtype=float)

            new_img = self.transform(new_img)
            # label_14 = self.label_14_transform(new_label)
            # label_28 = self.label_28_transform(new_label)
            # label_56 = self.label_56_transform(new_label)
            # label_112 = self.label_112_transform(new_label)
            label_224 = self.t_transform(new_label)
            # new_label2 = self.t_transform(new_label2)

            sam_image = self.sam_transform.apply_image(new_img2)
            input_image_torch = self.transform(sam_image)

            # input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

            return new_img, label_224, input_image_torch
        else:
            
            new_img = trans.Scale((random_size, random_size))(image)
            new_label = trans.Scale((random_size, random_size), interpolation=Image.NEAREST)(label)
            new_img2 = np.asarray(new_img, dtype=float)

            sam_image = self.sam_transform.apply_image(new_img2)
            input_image_torch = torch.as_tensor(sam_image).permute(2,0,1).contiguous()
            # print(input_image_torch.shape)
            # input_image_torch = self.t_transform(sam_image)

            image = self.transform(image)
            label_256 = self.t_transform(new_label)

            return image, input_image_torch, image_w, image_h, self.image_path[item], label_256

    def __len__(self):
        return len(self.image_path)


def get_loader(dataset_list, data_root, img_size, mode='train'):

    if mode == 'train':

        transform = trans.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        t_transform = trans.Compose([
            transforms.ToTensor(),
        ])
        # label_14_transform = trans.Compose([
        #     trans.Scale((img_size // 16, img_size // 16), interpolation=Image.NEAREST),
        #     transforms.ToTensor(),
        # ])
        # label_28_transform = trans.Compose([
        #     trans.Scale((img_size//8, img_size//8), interpolation=Image.NEAREST),
        #     transforms.ToTensor(),
        # ])
        # label_56_transform = trans.Compose([
        #     trans.Scale((img_size//4, img_size//4), interpolation=Image.NEAREST),
        #     transforms.ToTensor(),
        # ])
        # label_112_transform = trans.Compose([
        #     trans.Scale((img_size//2, img_size//2), interpolation=Image.NEAREST),
        #     transforms.ToTensor(),
        # ])
        scale_size = 256
    else:
        t_transform = trans.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        transform = trans.Compose([
            trans.Scale((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 处理的是Tensor
        ])
        scale_size = 256

    if mode == 'train':
        dataset = ImageData(dataset_list, data_root, transform, mode, img_size, scale_size, t_transform)
    else:
        dataset = ImageData(dataset_list, data_root, transform, mode, img_size, scale_size, t_transform)

    # data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_thread)
    return dataset