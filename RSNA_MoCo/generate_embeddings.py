from __future__ import print_function
import numpy as np
from sklearn.metrics.ranking import roc_auc_score
import os
import sys
import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import argparse
from random import shuffle
import socket
from sklearn.metrics.ranking import roc_auc_score
from dataset import RSNA_Data_finetune
import torch.multiprocessing as mp
import torch.distributed as dist
from torchvision import transforms, datasets
from util import adjust_learning_rate, AverageMeter
from sklearn.metrics import log_loss
from models.resnet import InsResNet50
from models.LinearModel import LinearClassifierResNet
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter

from sklearn import datasets
import pdb
from sklearn.manifold import TSNE
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def computeAUC(dataGT, dataPRED, classCount):
    outAUROC = []
    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()
    for i in range(classCount):
        outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
    mean_auc = float(np.mean(np.array(outAUROC)))
    return outAUROC, round(mean_auc, 4)


def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int,
                        default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int,
                        default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int,
                        default=5000, help='save frequency')
    parser.add_argument('--batch_size', type=int,
                        default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int,
                        default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float,
                        default=1, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str,
                        default='10,15,20', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float,
                        default=0.2, help='decay rate for learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float,
                        default=0, help='weight decay')
    parser.add_argument('--beta1', type=float,
                        default=0.5, help='beta1 for Adam')
    parser.add_argument('--beta2', type=float,
                        default=0.999, help='beta2 for Adam')

    # model definition
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet50', 'resnet50x2', 'resnet50x4'])
    parser.add_argument('--model_path', type=str,
                        default='/home/jason/github/MIRL/RSNA_MoCo/MoCo_supervised_1percentage_data.pth', help='the model to test')
    parser.add_argument('--features_path', type=str,
                        default="./all_features_un.npy")
    parser.add_argument('--labels_path', type=str,
                        default="./all_labels_un.npy")
    parser.add_argument('--layer', type=int, default=7,
                        help='which layer to evaluate')

    # crop
    parser.add_argument('--crop', type=float, default=0.2, help='minimum crop')

    # dataset
    parser.add_argument('--train_txt', type=str,
                        default="../experiments_configure/train20F.txt")
    parser.add_argument('--val_txt', type=str,
                        default="../experiments_configure/valF.txt")
    parser.add_argument('--dataset', type=str, default='imagenet100',
                        choices=['imagenet100', 'imagenet'])
    parser.add_argument('--data_folder', type=str, default='/DATA2/Data/RSNA')
    parser.add_argument('--save_path', type=str, default='.')
    parser.add_argument('--tb_path', type=str, default='./ts_bd')
    # augmentation
    parser.add_argument('--aug', type=str, default='CJ',
                        choices=['NULL', 'CJ'])
    # add BN
    parser.add_argument('--bn', action='store_true',
                        help='use parameter-free BN')
    parser.add_argument('--cosine', action='store_true',
                        help='use cosine annealing')
    parser.add_argument('--adam', action='store_true',
                        help='use adam optimizer')
    # warmup
    parser.add_argument('--warm', action='store_true',
                        help='add warm-up setting')
    parser.add_argument('--amp', action='store_true',
                        help='using mixed precision')
    parser.add_argument('--opt_level', type=str,
                        default='O2', choices=['O1', 'O2'])
    parser.add_argument('--syncBN', action='store_true',
                        help='enable synchronized BN')
    # GPU setting
    parser.add_argument('--gpu', default='0', type=int, help='GPU id to use.')

    opt = parser.parse_args()

    if opt.dataset == 'imagenet':
        if 'alexnet' not in opt.model:
            opt.crop = 0.08

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = opt.model_path.split('/')[-2]
    opt.model_name = '{}_bsz_{}_lr_{}_decay_{}_crop_{}'.format(opt.model_name, opt.batch_size, opt.learning_rate,
                                                               opt.weight_decay, opt.crop)

    if opt.amp:
        opt.model_name = '{}_amp_{}'.format(opt.model_name, opt.opt_level)

    opt.model_name = '{}_aug_{}'.format(opt.model_name, opt.aug)

    if opt.bn:
        opt.model_name = '{}_useBN'.format(opt.model_name)
    if opt.adam:
        opt.model_name = '{}_useAdam'.format(opt.model_name)
    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    opt.tb_folder = os.path.join(
        opt.tb_path, opt.model_name + '_layer{}'.format(opt.layer))
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.save_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    opt.n_label = 6

    return opt


def validate_multilabel(val_loader, model, opt):
    model.eval()
    outGT = torch.FloatTensor().cuda()
    all_features = []
    targets = []
    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):
            input = input.float()
            if opt.gpu is not None:
                input = input.cuda(opt.gpu, non_blocking=True)
            input = input.float()
            target = target.view(-1,
                                 6).contiguous().cuda(non_blocking=True).float()
            outGT = torch.cat((outGT, target), 0)
            # compute output
            feat = model(input, opt.layer)
            feat = feat.detach().cpu().numpy()
            # print(feat.shape)
            all_features.append(feat)
            targets.append(target.cpu().numpy())
    return all_features, targets


features_path = "./all_features_un.npy"
labels_path = "./all_labels_un.npy"
low = 0
high = 700
class_Name = ['epidural', 'intraparenchymal',
              'intraventricular', 'subarachnoid', 'subdural', 'any']


def get_rsna_model():
    all_features = []
    all_labels = []
    features = np.load(features_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)
    print(labels.shape)
    for batch in features:
        all_features += batch.tolist()
    for batch in labels:
        all_labels += batch.tolist()
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    all_features = np.reshape(all_features, (-1, 128))
    all_labels = np.reshape(all_labels, (-1, 6))
    return all_features, np.argmax(all_labels, axis=1), all_labels.shape[0], all_labels.shape[1]


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    df = pd.DataFrame(data[:, 0], columns=['Dimension 1'])
    df['Dimension 2'] = data[:, 1]
    counts = Counter(label)
    df['Label'] = [class_Name[i] +
                   " (" + str(counts[i]) + ")" for i in label.tolist()]
    num_Classes = len(counts.keys())
    # fig = plt.figure()
    # ax = plt.subplot(111)
    # for i in range(data.shape[0]):
    #     plt.text(data[i, 0], data[i, 1], str(int(label[i])),
    #              color=plt.cm.Set1(label[i]),
    #              fontdict={'weight': 'bold', 'size': 9})

    fig = plt.figure(figsize=(16, 10))
    plt.title(title)
    print("The number of classes is", num_Classes)
    print(Counter(label))
    g = sns.scatterplot(
        x='Dimension 1', y='Dimension 2',
        hue='Label',
        style="Label",
        palette=sns.color_palette("hls", num_Classes),
        data=df,
        legend="full",
        alpha=0.8
    )

    g.legend(loc='upper right')
    return fig


def removeLargestClass(data, label):
    rem = [i != 0 for i in label]
    data = data[rem]
    label = label[rem]
    return data, label


def main():
    args = parse_option()
    #train_txt = "/media/ubuntu/data/train5F.txt"
    train_txt = args.train_txt
    val_txt = args.val_txt
    global best_acc1
    best_acc1 = 0
    lowest_loss = 100
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    # set the data loader
    train_folder = os.path.join(args.data_folder, 'RSNAFTR')
    val_folder = os.path.join(args.data_folder, 'RSNAFVAL')
    best_test_auc = 0
    image_size = 224
    crop_padding = 32
    mean = [0.5]
    std = [0.5]
    normalize = transforms.Normalize(mean=mean, std=std)

    if args.aug == 'NULL':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(args.crop, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    elif args.aug == 'CJ':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(args.crop, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        raise NotImplemented('augmentation not supported: {}'.format(args.aug))
    val_transform = transforms.Compose([
        transforms.Resize(image_size + crop_padding),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize])
    #train_txt = "/media/ubuntu/data/train5F.txt"
    f_train = open(train_txt)
    c_train = f_train.readlines()
    f_train.close()
    trainfiles = [s.replace('\n', '') for s in c_train]
    csv_label = "train.csv"
    train_dataset = RSNA_Data_finetune(
        trainfiles, csv_label, train_folder, train_transform)
    #val_txt = "/media/ubuntu/data/valF.txt"
    f_val = open(val_txt)
    c_val = f_val.readlines()
    f_val.close()
    valfiles = [s.replace('\n', '') for s in c_val]
    val_dataset = RSNA_Data_finetune(
        valfiles, csv_label, val_folder, val_transform)

    print("Labled data for training", len(train_dataset))
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(
            train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    # create model and optimizer
    if args.model == 'resnet50':
        model = InsResNet50()
        #classifier = LinearClassifierResNet(args.layer, args.n_label, 'avg', 1)
    elif args.model == 'resnet50x2':
        model = InsResNet50(width=2)
        classifier = LinearClassifierResNet(args.layer, args.n_label, 'avg', 2)
    elif args.model == 'resnet50x4':
        model = InsResNet50(width=4)
        classifier = LinearClassifierResNet(args.layer, args.n_label, 'avg', 4)
    else:
        raise NotImplementedError('model not supported {}'.format(args.model))

    print('==> loading pre-trained model')
    ckpt = torch.load(args.model_path)
    model.load_state_dict(ckpt['model'])
    print("==> loaded checkpoint '{}' (epoch {})".format(
        args.model_path, ckpt['epoch']))
    print('==> done')
    model = model.cuda()
    all_features, targets = validate_multilabel(val_loader, model, opt=args)
    np.save(args.features_path, all_features)
    np.save(args.labels_path, targets)

    print("Starts t-SNE_Plotting")
    model_name = args.model_path.split('/')[-1][:-4]
    data, label, n_samples, n_features = get_rsna_model()
    print(data.shape, label.shape)
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, perplexity=5, random_state=0, n_iter=2000)
    t0 = time.time()
    result = tsne.fit_transform(data)
    np.save("all_class.npy", result)
    result = np.load("all_class.npy")
    fig = plot_embedding(result, label,
                         't-SNE embedding of the last layer of encoder of ' + model_name + ' fine-tuning')
    # t-SNE embedding of the last layer of encoder of MoCo with 5% labeled data fine-tuning
    plt.savefig(model_name + ".pdf")
    print(time.time()-t0, "seconds")

    t0 = time.time()
    data, label = removeLargestClass(data, label)
    tsne = TSNE(n_components=2, perplexity=5, random_state=0, n_iter=2000)
    result = tsne.fit_transform(data)
    fig = plot_embedding(result, label,
                         't-SNE embedding of the last layer of encoder of ' + model_name + ' fine-tuning, underrepresented')
    plt.savefig(model_name + "_underrepresented.pdf")
    print(time.time()-t0, "seconds")


if __name__ == '__main__':
    main()
