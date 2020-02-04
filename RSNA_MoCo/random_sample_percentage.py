from __future__ import print_function

import numpy as np
from skimage import color
from PIL import Image
import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import os
import pandas as pd


class ImageFolderInstance(datasets.ImageFolder):
    """Folder datasets which ret                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            urns the index of the image as well
    """

    def __init__(self, root, transform=None, target_transform=None, two_crop=False):
        super(ImageFolderInstance, self).__init__(root, transform, target_transform)
        self.two_crop = two_crop

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        image = self.loader(path)
        if self.transform is not None:
            img = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.two_crop:
            img2 = self.transform(image)
            img = torch.cat([img, img2], dim=0)

        return img, target, index


class RGB2Lab(object):
    """Convert RGB PIL image to ndarray Lab."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2lab(img)
        return img


class RGB2HSV(object):
    """Convert RGB PIL image to ndarray HSV."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2hsv(img)
        return img


class RGB2HED(object):
    """Convert RGB PIL image to ndarray HED."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2hed(img)
        return img


class RGB2LUV(object):
    """Convert RGB PIL image to ndarray LUV."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2luv(img)
        return img


class RGB2YUV(object):
    """Convert RGB PIL image to ndarray YUV."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2yuv(img)
        return img


class RGB2XYZ(object):
    """Convert RGB PIL image to ndarray XYZ."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2xyz(img)
        return img


class RGB2YCbCr(object):
    """Convert RGB PIL image to ndarray YCbCr."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2ycbcr(img)
        return img


class RGB2YDbDr(object):
    """Convert RGB PIL image to ndarray YDbDr."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2ydbdr(img)
        return img


class RGB2YPbPr(object):
    """Convert RGB PIL image to ndarray YPbPr."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2ypbpr(img)
        return img


class RGB2YIQ(object):
    """Convert RGB PIL image to ndarray YIQ."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2yiq(img)
        return img


class RGB2CIERGB(object):
    """Convert RGB PIL image to ndarray RGBCIE."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2rgbcie(img)
        return img

class ACDC_Data(Dataset):
    """Folder datasets which returns the index of the image as well
    """

    def __init__(self, transform=None, target_transform=None, two_crop=False):
        super(ACDC_Data, self).__init__()
        self.two_crop = two_crop
        self.imgs = np.load("acdc_imgs.npy")
        self.labels = np.load("acdc_labels.npy")
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is class_index of the target class.
        """
        image = self.imgs[index]
        target = self.labels[index]
        image = Image.fromarray(image)
        target = Image.fromarray(target)
        if self.transform is not None:
            img = self.transform(image)
        '''if self.target_transform is not None:
            target = self.target_transform(target)'''

        if self.two_crop:
            img2 = self.transform(image)
            img = torch.cat([img, img2], dim=0)

        return img, index

    def __len__(self):
        return self.imgs.shape[2]


class RSNA_Data(Dataset):
    def __init__(self,
                 name_list,
                 train_png_dir,
                 transform):
        super(RSNA_Data, self).__init__()
        self.name_list = name_list
        self.transform = transform
        self.train_png_dir = train_png_dir

    def __getitem__(self, idx):

        filename = self.name_list[idx]
        filepath = os.path.join(self.train_png_dir, filename)
        image = Image.open(filepath)
        img = self.transform(image)


        img2 = self.transform(image)
        img = torch.cat([img, img2], dim=0)

        # print(label)
        # exit(0)

        return img

    def __len__(self):
        return len(self.name_list)

class RSNA_Data_finetune(Dataset):
    def __init__(self,
                 name_list,
                 label_csv,
                 transform):
        super(RSNA_Data_finetune, self).__init__()
        self.name_list = name_list
        self.transform = transform
        #self.train_png_dir = train_png_dir
        self.label_csv = pd.read_csv(label_csv)
        #print(self.label_csv.head())
        self.label_csv.set_index(['Image'], inplace=True)
        #print(self.label_csv.head())

    def __getitem__(self, idx):

        filename = self.name_list[idx]
        #filepath = os.path.join(self.train_png_dir, filename)
        #print(filepath)
        #image = Image.open(filepath)
        #img = self.transform(image)


        #img2 = self.transform(image)
        #img = torch.cat([img, img2], dim=0)
        labels = torch.tensor(
            self.label_csv.loc[
                filename[:-4], ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']])

        # print(label)
        # exit(0)
        return labels, filename

    def __len__(self):
        return len(self.name_list)

class RSNA_Data_submission(Dataset):
    def __init__(self,
                 name_list,
                 label_csv,
                 train_png_dir,
                 transform):
        super(RSNA_Data_submission, self).__init__()
        self.name_list = name_list
        self.transform = transform
        self.train_png_dir = train_png_dir
        self.label_csv = pd.read_csv(label_csv)
        #print(self.label_csv.head())
        self.label_csv.set_index(['Image'], inplace=True)
        #print(self.label_csv.head())

    def __getitem__(self, idx):

        filename = self.name_list[idx]
        filepath = os.path.join(self.train_png_dir, filename)
        #print(filepath)
        image = Image.open(filepath)
        img = self.transform(image)




        # print(label)
        # exit(0
        return img

    def __len__(self):
        return len(self.name_list)

def transfer_np_to_str_to_int(arr):
    str_label = ""
    #print(arr.shape)
    for a in range(arr.shape[1]):
        a = str(arr[0, a])
        #print(a)
        str_label += a
    #print(str_label)
    label_num = int(str_label, 2)
    return label_num

def text_save(filename, data):#filename为写入CSV文件的路径，data为要写入数据列表.
  file = open(filename, 'a')
  for i in range(len(data)):
    s = data[i]
    s = s +'\n'  #去除单引号，逗号，每行末尾追加换行符
    file.write(s)
  file.close()

if __name__ == '__main__':
    from torchvision import transforms
    from torch.utils.data import DataLoader
    import random
    #set the percentage 0.01 or 0.05
    percentage = 0.01
    train_txt = "./train" + str(percentage) + ".txt"
    name_file = "./experiments_configure/trainF.txt"
    #root = "F:/MICCAI 2020/unsupervised represent learning for segmentation/MoCo/tiny-imagenet-200/train/"
    mean = [0.5]
    std = [0.5]
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.2, 1.)),
        #transforms.RandomGrayscale(p=0.2),
        #transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    #name_file = "./experiments_configure/trainF.txt"
    f_train = open(name_file)
    c_train = f_train.readlines()
    f_train.close()
    name_file = [s.replace('\n', '') for s in c_train]
    csv_label = "./train.csv"
    #png_dir = "/media/ubuntu/data/RSNATR"
    dataset = RSNA_Data_finetune(name_file, csv_label, train_transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    label, filen = next(iter(loader))
    print(label.size())
    all_labels = []
    all_file_names = []
    for i, (labels, filename) in enumerate(loader):
        label_num = transfer_np_to_str_to_int(labels.numpy())
        #print(label_num)
        all_labels.append(str(label_num))
        all_file_names.append(filename)
    sign_labels = set(all_labels)
    print(sign_labels)
    all_things = {}
    for j, each_b in enumerate(sign_labels):
        all_things[each_b] = []
    for k, each_a in enumerate(all_labels):
        all_things[str(each_a)].append(all_file_names[k][0])
    print(all_things['7'])
    store_list = []
    for j, each_b in enumerate(sign_labels):
        each_b_files = all_things[each_b]
        random.shuffle(each_b_files)
        if len(each_b_files) < 100:
            select_file = [each_b_files[0]]
        else:
            select_file = each_b_files[:int(percentage*len(each_b_files))]
        for file_n in select_file:
            store_list.append(file_n)
    print(len(store_list))
    text_save(train_txt, store_list)
