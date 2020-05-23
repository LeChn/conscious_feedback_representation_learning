from __future__ import print_function
import os
import sys
import time
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import argparse
import socket
from dataset import RSNA_Data
import tensorboard_logger as tb_logger
import torchvision
from torchvision import transforms, datasets
from util import adjust_learning_rate, AverageMeter

from models.simCLR import simCLR
from models.deepInfoMax import DeepInfoMaxLoss
from NCE.NCEAverage import MemoryInsDis
from NCE.NCEAverage import MemoryMoCo
from NCE.NCECriterion import NCECriterion
from NCE.NCECriterion import NCESoftmaxLoss
import pdb
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def parse_option():

    #hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=20, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=18, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='100,120,200', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.25, help='decay rate for learning rate')
    parser.add_argument('--beta1', type=float,  default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # crop
    parser.add_argument('--crop', type=float, default=0.4, help='minimum crop')

    # dataset
    parser.add_argument('--train_txt', type=str, default="../experiments_configure/train100F.txt")
    parser.add_argument('--dataset', type=str, default='imagenet100', choices=['imagenet100', 'imagenet'])
    parser.add_argument('--data_folder', type=str, default="/DATA2/Data/RSNA/RSNAFTR")

    # resume
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')

    # augmentation setting
    parser.add_argument('--aug', type=str, default='NULL', choices=['NULL', 'CJ'])

    # warm up
    parser.add_argument('--warm',  default=False, help='add warm-up setting')
    parser.add_argument('--amp',  default=False, help='using mixed precision')
    parser.add_argument('--opt_level', type=str, default='O2', choices=['O1', 'O2'])

    # model definition
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'resnet50x2', 'resnet50x4'])
    parser.add_argument('--model_path', type=str, default='.')
    parser.add_argument('--tb_path', type=str, default='.')
    # loss function
    parser.add_argument('--softmax', default=True, help='using softmax contrastive loss rather than NCE')
    parser.add_argument('--nce_k', type=int, default=16384)
    parser.add_argument('--nce_t', type=float, default=0.07)
    parser.add_argument('--nce_m', type=float, default=0.5)

    # memory setting
    parser.add_argument('--moco', default=True, help='using MoCo (otherwise Instance Discrimination)')
    parser.add_argument('--alpha', type=float, default=0.999, help='exponential moving average weight')

    # GPU setting
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.method = 'softmax' if opt.softmax else 'nce'
    prefix = 'MoCo{}'.format(opt.alpha) if opt.moco else 'InsDis'

    opt.model_name = '{}_{}_{}_{}_lr_{}_decay_{}_bsz_{}_crop_{}'.format(prefix, opt.method, opt.nce_k, opt.model, opt.learning_rate, opt.weight_decay, opt.batch_size, opt.crop)
    opt.model_name = '{}_aug_{}'.format(opt.model_name, opt.aug)

    opt.model_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.model_folder):
        os.makedirs(opt.model_folder)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    return opt


def main():

    args = parse_option()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, pin_memory=True, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # create model and optimizer
    n_data = len(trainset)

    # Resnet50 with MLP projection head
    model = simCLR(in_channel=3)
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    loss_fn = DeepInfoMaxLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    args.start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            # checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            print(f"=> loaded successfully '{args.resume}' (epoch {checkpoint['epoch']})")
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    # tensorboard
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        loss = train_cifar(epoch, train_loader, model, loss_fn, optimizer, args)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('ins_loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # save model
        if epoch % args.save_freq == 0 and epoch > args.save_freq:
            print('==> Saving...')
            state = {
                'opt': args,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
            # help release GPU memory
            del state

        # saving the model
        print('==> Saving...')
        state = {
            'opt': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
        save_file = os.path.join(args.model_folder, 'current.pth')
        #torch.save(state, save_file)
        if epoch % args.save_freq == 0:
            save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
        # help release GPU memory
        del state
        torch.cuda.empty_cache()


def train_cifar(epoch, train_loader, model, loss_fn, optimizer, opt):
    """
    one epoch training for instance discrimination
    """
    print("Train Cifar with InfoMAX!")
    model.train()

    def set_bn_train(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

    end = time.time()
    for idx, (inputs, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = inputs.size(0)

        inputs = inputs.float().cuda()

        # ===================forward=====================
        _, feat_q, M = model(inputs)
        M_prime = torch.cat((M[1:], M[0:1]), dim=0)
        loss = loss_fn(feat_q, M, M_prime)  # 9 + 2 -> 3 + 2
        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        loss_meter.update(loss.item(), bsz)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'.format(
                      epoch, idx + 1, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=loss_meter))
            # print(out.shape)
            sys.stdout.flush()

    return loss_meter.avg


if __name__ == '__main__':
    main()
