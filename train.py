import argparse
import logging
import time
import os
from copy import deepcopy

import torch
import torch.optim as optim
from torch.cuda import amp
from datetime import datetime
from tqdm import tqdm

from datasets import make_data_loader, GridMask
from models import create_model
from loss import BCELoss
from utils.metrics import AverageValueMeter, AveragePrecisionMeter, SumValueMeter
from utils.train_utils import save_checkpoint, init_seeds, adjust_learning_rate
from utils.ema import ModelEma

parser = argparse.ArgumentParser(
    description='PyTorch Training for Multi-label Image Classification with partial labels')

''' Fixed in general '''
parser.add_argument('--data_root_dir', default='./data/', type=str, help='data root path')
parser.add_argument('--image_size', '-i', default=448, type=int, help='input size')
parser.add_argument('--epochs', default=20, type=int, help='25 (VOC), 20 (others)')
parser.add_argument('--epoch_step', default=[15, 20], type=int, nargs='+', help='number of epochs to change learning rate')
parser.add_argument('-b', '--batch_size', default=32, type=int, help='batch size')
parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='INT',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--display_interval', default=100, type=int, metavar='M', help='display_interval')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--w_d', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--device', default='0', help='cuda device id, e.g. 0')
parser.add_argument('--amp', action='store_true', help='mixing precision, speed up training, degrade precision')

''' Train setting '''
parser.add_argument('--dataset', required=True, choices=['VOC2007', 'VG', 'COCO2014'], help='dataset name')
parser.add_argument('--model_name', type=str, default='Resnet101', help='Resnet101 or Resnet50')
parser.add_argument('--frozen_stage', type=int, default=1, help='frozen stage of ResNet i.e. -1 (not frozen),0,1,2,3,4')
parser.add_argument('--model_ema', action='store_true',
                    help='whether the model adopts exponential moving average method')
parser.add_argument('--ema_decay', type=float, default=0.9997)
parser.add_argument('--save_dir', default='./checkpoint/', type=str, help='save path')
parser.add_argument('--sample_partial_mode', default='None', type=str, metavar='MODE',
                    help='sample mode of label-observe-matrix, including class and img')
parser.add_argument('--sample_partial_param', default=[0.1], nargs='+',
                    help='observed number of labels, it can be (0,1], -1 or int(>=0)')

''' hyperparameters '''
parser.add_argument('--topk', type=int, default=5)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--tau', type=float, default=0.6)
parser.add_argument('--lambda_u', type=float, default=1.0, help='weight of the pseudo-label loss')
parser.add_argument('--noSR', action='store_true', default=False)

''' Val or Test setting '''
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='checkpoint to evaluate (with --evaluate) or to continue training from')


def main():
    global best_score
    global logger
    args = parser.parse_args()

    format = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s', '%Y-%m-%d %H.%M.%S')
    logger = logging.getLogger(__name__)

    log_dir = os.path.join('log', args.dataset)
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(f"{log_dir}/{datetime.now().strftime('%Y-%m-%d %H.%M.%S')}_"
                                       f"{args.epochs}_{args.batch_size}_{args.num_workers}_{args.lr}_{args.amp}_"
                                       f"{args.model_ema}_{args.tau}_{args.lambda_u}_{args.sample_partial_mode}_{args.sample_partial_param}.log")
    console = logging.StreamHandler()

    file_handler.setFormatter(format)
    console.setFormatter(format)

    logger.addHandler(file_handler)
    logger.addHandler(console)
    logger.setLevel(logging.DEBUG)

    logger.info('---------------Args Items--------------')
    for k, v in vars(args).items():
        logger.info('{}: {}'.format(k, v))
    logger.info('---------------Args Items--------------\n')

    if args.resume and not os.path.isfile(args.resume):
        raise SystemExit(f'--resume: no such file {args.resume}')
    if args.evaluate and not args.resume:
        raise SystemExit('--evaluate needs --resume path/to/checkpoint.pth')

    init_seeds(0)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    # evaluation needs only the test split, so it skips building (and downloading) the training set
    train_loader, test_loader, num_classes = make_data_loader(args, is_train=not args.evaluate)
    args.num_classes = num_classes
    args.class_th = (args.tau, args.tau)
    args.grid = GridMask(128, 448, rotate=360, ratio=0.6, mode=1, prob=0.8)
    model = create_model(args).to(device)
    if args.amp:
        model.half().float()

    start_epoch = 0
    end_epoch = args.epochs
    best_score = 0

    criterion = {'BCELoss': BCELoss(reduce=True, size_average=True).to(device)}

    checkpoint = None
    if args.resume:
        logger.info("* Loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=device)
        start_epoch = checkpoint['epoch']
        best_score = checkpoint['best_score']
        logger.info("==> Checkpoint Epoch: {0}, mAP: {1}".format(start_epoch, best_score))

    if args.evaluate:
        # the EMA weights are the ones validated during training
        model.load_state_dict(checkpoint['ema_state_dict'] if args.model_ema else checkpoint['state_dict'])
        eva_model = model.half() if args.amp else model
        validate(test_loader, eva_model, criterion, device, args, 10000)
        return

    if checkpoint is not None:  # continue training: the live weights first, so the EMA copy starts from them
        model.load_state_dict(checkpoint['state_dict'])

    ema_model = None
    if args.model_ema:
        ema_model = ModelEma(model, decay=args.ema_decay, device=device)
        if checkpoint is not None and 'ema_state_dict' in checkpoint:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])

    backbone_base, backbone_layer4, new_params = model.trainable_parameters()
    optimizer = optim.SGD(
        [
            {'params': backbone_base, 'lr': args.lr * 0.1},
            {'params': backbone_layer4, 'lr': args.lr * 0.5},
            {'params': new_params, 'lr': args.lr}
        ],
        momentum=args.momentum, weight_decay=args.w_d)
    if checkpoint is not None:
        if 'optimizer' in checkpoint:  # also restores the learning rates already decayed
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:                          # older checkpoints: replay the decays that have already happened
            for step in args.epoch_step:
                if step < start_epoch:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.1

    for epoch in range(start_epoch, end_epoch):
        lr_now = adjust_learning_rate(args, epoch, optimizer)
        args.grid.set_prob(epoch, 10)

        train(args, train_loader, model, ema_model, criterion, optimizer, epoch, device)

        if ema_model is not None:
            test_model = deepcopy(ema_model.ema).half() if args.amp else deepcopy(ema_model.ema)
        else:
            test_model = model

        score = validate(test_loader, test_model, criterion, device, args, epoch)

        is_best = score > best_score
        best_score = max(score, best_score)
        checkpoint = {
            'epoch': epoch + 1,
            'model_name': args.model_name,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_score': best_score,
            'alpha': args.alpha,
            'topk': args.topk,
            'image_size': args.image_size,
        }
        if args.model_ema:
            checkpoint['ema_state_dict'] = test_model.state_dict()
        model_dir = args.save_dir
        save_checkpoint(checkpoint, model_dir, is_best, epoch)
        logger.info(' * best mAP={best:.4f}'.format(best=best_score))


def train(args, train_loader, model, ema_model, criterion, optimizer, epoch, device):
    model.train()

    sup_loss_meter = AverageValueMeter('sup_loss')
    unsup_loss_meter = AverageValueMeter('unsup_loss')
    loss_meter = AverageValueMeter('loss')
    data_time = AverageValueMeter('data_time')
    batch_time = AverageValueMeter('batch_time')
    num_meter = SumValueMeter(args.num_classes)

    step = 0
    scaler = amp.GradScaler(enabled=args.amp)

    st_time = time.time()
    for i, (inputs, targets) in enumerate(train_loader):
        dt_time = time.time() - st_time
        data_time.add(dt_time)

        aug_w, aug_s = inputs[0]
        targets = targets.float().to(device)
        bs = aug_w.shape[0]

        pos_th, neg_th = args.class_th[0], args.class_th[1]

        aug_s = args.grid(aug_s)
        imgs = torch.cat((aug_w, aug_s), dim=0).to(device)

        with amp.autocast(enabled=args.amp):
            outputs = model(imgs)

            if isinstance(outputs, tuple):
                logits, class_features = outputs
            else:
                logits = outputs

            logits_w = logits[:bs]
            logits_s = logits[bs:]

            pseudo_hard_labels = unlabel_pseudo_labeling(logits_w, targets, pos_th, neg_th)

            p_n = (pseudo_hard_labels == 1).sum(0)
            n_n = (pseudo_hard_labels == -1).sum(0)
            num_meter.add(p_n.cpu(), n_n.cpu())

            sup_loss = criterion['BCELoss'](logits_w, targets) + criterion['BCELoss'](logits_s, targets)

            unsup_loss = criterion['BCELoss'](logits_s, pseudo_hard_labels.float())

            loss = sup_loss + args.lambda_u * unsup_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if ema_model is not None:
            ema_model.update(model)

        loss_meter.add(loss.item(), imgs.size(0))
        sup_loss_meter.add(sup_loss.item(), imgs.size(0))
        unsup_loss_meter.add(unsup_loss.item() if isinstance(unsup_loss, torch.Tensor) else unsup_loss,
                             imgs.size(0))

        bt_time = time.time() - st_time
        batch_time.add(bt_time)

        st_time = time.time()
        step += 1

        if i % args.display_interval == 0:
            logger.info('{0}\t''Epoch: [{1}] [{2}/{3}]\t'
                        'Batch_Time: {batch_time_cur:.3f} ({batch_time:.3f})\t'
                        'Data_Time: {data_time_cur:.3f} ({data_time:.3f})\t'
                        'Loss: {loss_cur:.4f} ({loss:.4f})\t'
                        'SupLoss: {loss1_cur:.4f} ({loss1:.4f})\t'
                        'UnsupLoss: {loss2_cur:.4f} ({loss2:.4f})\t'
                        .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, i, len(train_loader),
                                batch_time_cur=batch_time.value(), batch_time=batch_time.average(),
                                data_time_cur=data_time.value(), data_time=data_time.average(),
                                loss_cur=loss_meter.value(), loss=loss_meter.average(),
                                loss1_cur=sup_loss_meter.value(), loss1=sup_loss_meter.average(),
                                loss2_cur=unsup_loss_meter.value(), loss2=unsup_loss_meter.average()
                                ))

    loss = loss_meter.average()
    logger.info('* Train-Epoch:[{0}]\t Loss: {loss:.4f}\t '.format(epoch, loss=loss))


def unlabel_pseudo_labeling(logits_w, targets, pos_th, neg_th):
    pseudo_hard_labels = torch.zeros_like(targets)
    with torch.no_grad():
        unlabel_part_mask = (targets == 0)
        p_pos = torch.sigmoid(logits_w.detach())
        p_neg = 1 - p_pos
        pos_mask = (p_pos * unlabel_part_mask) >= pos_th
        neg_mask = (p_neg * unlabel_part_mask) >= neg_th
        pseudo_hard_labels[pos_mask] = 1
        pseudo_hard_labels[neg_mask] = -1
    return pseudo_hard_labels


@torch.no_grad()
def validate(val_loader, model, criterion, device, args, epoch):
    model.eval()

    ap_meter = AveragePrecisionMeter()
    loss_meter = AverageValueMeter('loss')
    data_time = AverageValueMeter('data_time')
    batch_time = AverageValueMeter('batch_time')

    st_time = time.time()
    val_loader = tqdm(val_loader, desc='Validate')

    for i, (inputs, targets) in enumerate(val_loader):
        dt_time = time.time() - st_time
        data_time.add(dt_time)

        imgs = inputs[0].half().to(device) if args.amp else inputs[0].to(device)
        targets = targets.float().to(device)

        outputs = model(imgs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        loss = criterion['BCELoss'](outputs, targets)
        loss_meter.add(loss.item(), imgs.size(0))

        bt_time = time.time() - st_time
        batch_time.add(bt_time)

        ap_meter.add(outputs.cpu().data, targets.cpu())
        st_time = time.time()

    ap = ap_meter.value()
    val_map = ap.mean()


    OP, OR, OF1, CP, CR, CF1 = ap_meter.overall()
    OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = ap_meter.overall_topk(3)

    logger.info('* Test\nLoss: {loss:.4f}\t mAP: {map:.4f}\t'
                'Data_time: {data_time:.4f}\t Batch_time: {batch_time:.4f}'.format(
        loss=loss_meter.average(), map=val_map, data_time=data_time.average(), batch_time=batch_time.average()))
    logger.info('OP: {OP:.3f}\t OR: {OR:.3f}\t OF1: {OF1:.3f}\t'
                'CP: {CP:.3f}\t CR: {CR:.3f}\t CF1: {CF1:.3f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
    logger.info('OP_3: {OP:.3f}\t OR_3: {OR:.3f}\t OF1_3: {OF1:.3f}\t'
                'CP_3: {CP:.3f}\t CR_3: {CR:.3f}\t CF1_3: {CF1:.3f}'.format(OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k,
                                                                            CR=CR_k, CF1=CF1_k))
    return val_map


if __name__ == "__main__":
    main()
