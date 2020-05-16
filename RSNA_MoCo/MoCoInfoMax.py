import numpy as np
from sklearn.metrics.ranking import roc_auc_score
import os
import sys
import time
import itertools
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
from models.simCLR import simCLR
from models.LinearModel import LinearClassifierResNet
from models.deepInfoMax import GlobalDiscriminator, LocalDiscriminator, PriorDiscriminator, DeepInfoMaxLoss
import tensorboard_logger as tb_logger
import pdb
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def computeAUC(dataGT, dataPRED, classCount):
    outAUROC = []
    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()
    for i in range(classCount):
        outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
    mean_auc = float(np.mean(np.array(outAUROC)))
    return outAUROC, round(mean_auc, 4)


def parse_option():

    # hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=5000, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.03, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='100,120', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')

    # model definition
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'resnet50x2', 'resnet50x4'])
    parser.add_argument('--model_path', type=str, default='/home/jason/github/MIRL/RSNA_MoCo/MoCoV2/ckpt_epoch_200.pth', help='the model to test')
    parser.add_argument('--layer', type=int, default=6, help='which layer to evaluate')
    parser.add_argument('--resnet', action='store_true', help='ignore pre-trained encoder')
    parser.add_argument('--freeze', action='store_true', help='freeze the encoder')

    # crop
    parser.add_argument('--crop', type=float, default=0.2, help='minimum crop')

    # dataset
    parser.add_argument('--train_txt', type=str, default="../experiments_configure/train100F.txt")
    parser.add_argument('--val_txt', type=str, default="../experiments_configure/valF.txt")
    parser.add_argument('--dataset', type=str, default='imagenet100', choices=['imagenet100', 'imagenet'])
    parser.add_argument('--data_folder', type=str, default='/DATA2/Data/RSNA')
    parser.add_argument('--save_path', type=str, default='/home/jason/github/MIRL/RSNA_MoCo/finetunemodel')
    parser.add_argument('--tb_path', type=str, default='/home/jason/github/MIRL/RSNA_MoCo/ts_bd')
    # augmentation
    parser.add_argument('--aug', type=str, default='CJ', choices=['NULL', 'CJ'])
    # add BN
    parser.add_argument('--bn', action='store_true', help='use parameter-free BN')
    parser.add_argument('--cosine', action='store_true', help='use cosine annealing')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    # warmup
    parser.add_argument('--warm', action='store_true', help='add warm-up setting')
    parser.add_argument('--amp', action='store_true', help='using mixed precision')
    parser.add_argument('--opt_level', type=str, default='O2', choices=['O1', 'O2'])
    parser.add_argument('--syncBN', action='store_true', help='enable synchronized BN')
    # GPU setting
    parser.add_argument('--gpu', default='0', type=int, help='GPU id to use.')

    opt = parser.parse_args()
    opt.lr_decay_epochs = [int(learn_rate) for learn_rate in opt.lr_decay_epochs.split(',')]
    opt.model_name = f'{"ResNet" if opt.resnet else "MoCoV2_Freeze" if opt.freeze else "MoCoV2"}_{opt.batch_size}_bsz_{opt.learning_rate}_lr_{opt.weight_decay}_decay_{opt.crop}_crop'

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name + '_layer{}'.format(opt.layer))
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.save_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    opt.n_label = 6

    return opt


def main():
    args = parse_option()
    train_txt = args.train_txt
    val_txt = args.val_txt
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
    val_dataset = RSNA_Data_finetune(valfiles, csv_label, val_folder, val_transform)

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
        model = simCLR()
        classifier = LinearClassifierResNet(args.layer, args.n_label, 'avg', 1)
    elif args.model == 'resnet50x2':
        model = simCLR()
        classifier = LinearClassifierResNet(args.layer, args.n_label, 'avg', 2)
    elif args.model == 'resnet50x4':
        model = simCLR()
        classifier = LinearClassifierResNet(args.layer, args.n_label, 'avg', 4)
    else:
        raise NotImplementedError('model not supported {}'.format(args.model))

    if not args.resnet:
        print('==> loading pre-trained model')
        ckpt = torch.load(args.model_path)
        model.load_state_dict(ckpt['model'])
        print("==> loaded checkpoint '{}' (epoch {})".format(
            args.model_path, ckpt['epoch']))
        print('==> done')

    model = model.cuda()
    classifier = classifier.cuda()

    #criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)
    criterion = torch.nn.BCEWithLogitsLoss().cuda()
    loss_fn = DeepInfoMaxLoss().cuda()

    if args.freeze:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(itertools.chain(model.parameters(), classifier.parameters()), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    model.eval()
    cudnn.benchmark = True
    args.start_epoch = 1

    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)
    for epoch in range(args.start_epoch, args.epochs + 1):
        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        train_auc = train(epoch, train_loader, model, classifier, criterion, loss_fn, optimizer, args)
        time2 = time.time()
        print('train epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        print("==> testing...")
        auc, mean_auc, test_loss = validate_multilabel(val_loader, model, classifier, criterion, args)
        if mean_auc > best_test_auc:
            best_test_auc = mean_auc
        # save the best model
        '''if test_loss < lowest_loss:
            lowest_loss = test_loss
            print('saving best model!')
            torch.save(classifier.state_dict(), "best_classifier.pth")'''
        logger.log_value('mean_auc', mean_auc, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        logger.log_value('best_auc', best_test_auc, epoch)
        print('best test auc is:', best_test_auc)
        pass


def set_lr(optimizer, lr):
    """
    set the learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(epoch, train_loader, model, classifier, criterion, loss_fn, optimizer, opt):
    """
    one epoch training
    """

    if opt.freeze:
        """
        Switch to eval mode:
        Under the protocol of linear classification on frozen features/models,
        it is not legitimate to change any part of the pre-trained model.
        BatchNorm in train mode may revise running mean/std (even if it receives
        no gradient), which are part of the model parameters too.
        """
        model.eval()
    else:
        model.train()
    classifier.train()
    outGT = torch.FloatTensor().cuda()
    outPRED = torch.FloatTensor().cuda()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if opt.gpu is not None:
            input = input.cuda(opt.gpu, non_blocking=True)
        input = input.float()
        target = target.float()
        target = target.view(-1, 6).contiguous().cuda(async=True)
        outGT = torch.cat((outGT, target), 0)

        # ===================forward=====================
        if opt.freeze:
            with torch.no_grad():
                feat, _, M = model(input)
                feat = feat.detach()
        else:
            feat, _, M = model(input)

        M_prime = torch.cat((M[1:], M[0:1]), dim=0)
        output = classifier(feat)
        outPRED = torch.cat((outPRED, output.data), 0)
        loss = criterion(output, target.float()) + loss_fn(output, M, M_prime)
        losses.update(loss.item(), input.size(0))
        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      epoch, idx, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses))
            #print('TRAIN Logloss is {}, mean_Logloss is {}'.format(Logloss, mean_ll))
            sys.stdout.flush()
    auc_train, mean_auc_train = computeAUC(outGT, outPRED, 6)
    print('All Train AUC is {},  Mean_AUC IS {}'.format(auc_train, mean_auc_train))
    return mean_auc_train


def validate_multilabel(val_loader, model, classifier, criterion, opt):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode, linear classification on frozen features/models,
    model.eval()
    classifier.eval()
    outGT = torch.FloatTensor().cuda()
    outPRED = torch.FloatTensor().cuda()
    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):
            input = input.float()
            if opt.gpu is not None:
                input = input.cuda(opt.gpu, non_blocking=True)
            input = input.float()
            target = target.view(-1, 6).contiguous().cuda(async=True).float()
            outGT = torch.cat((outGT, target), 0)
            # compute output
            feat = model(input)[0]
            feat = feat.detach()
            output = classifier(feat)
            outPRED = torch.cat((outPRED, output.data), 0)
            loss = criterion(output, target.float())
            losses.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(idx, len(val_loader), batch_time=batch_time, loss=losses))
                #print('Test Logloss is {},mean_Logloss is {:.3f}'.format(Logloss, mean_ll))
    auc_test, mean_auc_test = computeAUC(outGT, outPRED, 6)
    '''auc = [round(x, 4) for x in auc]
    Logloss = [round(x, 4) for x in Logloss]'''
    print('All Test AUC is {}, Mean_AUC IS {}'.format(auc_test, mean_auc_test))
    return auc_test, mean_auc_test, losses.avg


if __name__ == '__main__':
    main()
