"""Train CIFAR-10 from local class folders."""
import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from models import *
from utils import progress_bar


model_factories = {
    'ResNet18': ResNet18,
    'GoogLeNet': GoogLeNet,
    'DPN92': DPN92,
    'ShuffleNetV2': lambda: ShuffleNetV2(1),
    'EfficientNetB0': EfficientNetB0,
    'RegNetX_200MF': RegNetX_200MF,
    'SimpleDLA': SimpleDLA,
    'DLA': DLA
}


def mixup_batch(inputs, targets, alpha):
    if alpha <= 0:
        return inputs, targets, targets, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    indices = torch.randperm(inputs.size(0), device=inputs.device)
    mixed = lam * inputs + (1.0 - lam) * inputs[indices]
    return mixed, targets, targets[indices], lam


def train(net, loader, criterion, optimizer, device, mixup_alpha):
    net.train()
    loss_sum = correct = total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets_a, targets_b, lam = mixup_batch(inputs, targets, mixup_alpha)
        optimizer.zero_grad(set_to_none=True)
        outputs = net(inputs)
        loss = (lam * criterion(outputs, targets_a) +
                (1.0 - lam) * criterion(outputs, targets_b))
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        loss_sum += loss.item() * batch_size
        predicted = outputs.argmax(dim=1)
        correct += (lam * predicted.eq(targets_a).sum().item() +
                    (1.0 - lam) * predicted.eq(targets_b).sum().item())
        total += batch_size
        # With Mixup, this is weighted accuracy against both source labels.
        progress_bar(batch_idx, len(loader),
                     'Loss: %.3f | %s: %.3f%%' %
                     (loss_sum / total, 'Mixup Acc' if mixup_alpha > 0 else 'Acc',
                      100.0 * correct / total))
    return loss_sum / total, 100.0 * correct / total


@torch.no_grad()
def test(net, loader, criterion, device):
    net.eval()
    loss_sum = correct = total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        batch_size = targets.size(0)
        loss_sum += criterion(outputs, targets).item() * batch_size
        correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total += batch_size
        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                     (loss_sum / total, 100.0 * correct / total, correct, total))
    return loss_sum / total, 100.0 * correct / total


def load_weights(net, state_dict):
    # Allow checkpoints saved with DataParallel to load on a single GPU or CPU.
    model = net.module if isinstance(net, nn.DataParallel) else net
    model.load_state_dict({key.removeprefix('module.'): value
                           for key, value in state_dict.items()})


def build_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--arch', default='ResNet18', choices=sorted(model_factories))
    parser.add_argument('--run-name', help='unique experiment name; required with --resume')
    parser.add_argument('--data-root', default='/home/wzc/nas/home/Dataset/CIFAR10/CIFAR10')
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--epochs', default=300, type=int, help='total epochs, including resumed epochs')
    parser.add_argument('--warmup-epochs', default=5, type=int)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--workers', default=2, type=int)
    parser.add_argument('--auto-augment', default='cifar10', choices=['none', 'cifar10'])
    parser.add_argument('--mixup-alpha', default=0.2, type=float, help='0 disables Mixup')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--resume', '-r', action='store_true', help='resume last.pth in the selected run')
    group.add_argument('--init-from', help='initialize weights from an existing checkpoint in a NEW run')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.resume and not args.run_name:
        parser.error('--resume requires --run-name')
    if args.run_name and (args.run_name in ('.', '..') or
                          os.path.basename(args.run_name) != args.run_name):
        parser.error('--run-name must be a single directory name')
    if args.epochs <= 0 or not 0 <= args.warmup_epochs < args.epochs:
        parser.error('require epochs > 0 and 0 <= warmup-epochs < epochs')
    if args.lr <= 0 or args.mixup_alpha < 0 or args.weight_decay < 0:
        parser.error('require lr > 0, mixup-alpha >= 0 and weight-decay >= 0')
    if args.batch_size <= 0 or args.workers < 0:
        parser.error('require batch-size > 0 and workers >= 0')

    run_name = args.run_name or datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    checkpoint_dir = os.path.join('checkpoint', args.arch, run_name)
    best_path = os.path.join(checkpoint_dir, 'ckpt.pth')
    last_path = os.path.join(checkpoint_dir, 'last.pth')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = None
    if args.resume:
        if not os.path.isfile(last_path):
            parser.error('no last.pth; for an older weights-only checkpoint, use --init-from PATH in a new run')
        checkpoint = torch.load(last_path, map_location=device, weights_only=True)
        # The saved schedule and augmentation settings define a resumed experiment.
        for key in ('lr', 'epochs', 'warmup_epochs', 'weight_decay', 'batch_size',
                    'auto_augment', 'mixup_alpha'):
            setattr(args, key, checkpoint['args'][key])
    else:
        os.makedirs(checkpoint_dir, exist_ok=False)
    print('==> Model: %s | Run: %s | Checkpoints: %s' %
          (args.arch, run_name, checkpoint_dir))
    print('==> epochs=%d, lr=%g, warmup=%d, auto_augment=%s, mixup_alpha=%g' %
          (args.epochs, args.lr, args.warmup_epochs, args.auto_augment, args.mixup_alpha))

    print('==> Preparing local data..')
    train_transforms = [transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip()]
    if args.auto_augment == 'cifar10':
        train_transforms.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose(train_transforms + [transforms.ToTensor(), normalize])
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])
    trainset = torchvision.datasets.ImageFolder(
        os.path.join(args.data_root, 'train'), transform=transform_train)
    testset = torchvision.datasets.ImageFolder(
        os.path.join(args.data_root, 'test'), transform=transform_test)
    if len(trainset.classes) != 10 or trainset.class_to_idx != testset.class_to_idx:
        raise ValueError('Expected matching train/test folders with 10 classes')
    print('==> Training images: %d | Test images: %d' % (len(trainset), len(testset)))
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device == 'cuda'))
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False,
        num_workers=args.workers, pin_memory=(device == 'cuda'))

    print('==> Building model..')
    net = model_factories[args.arch]().to(device)
    if device == 'cuda':
        net = nn.DataParallel(net)
        cudnn.benchmark = True
    if args.init_from:
        initial = torch.load(args.init_from, map_location=device, weights_only=True)
        load_weights(net, initial['net'])
        print('==> Initialized weights from %s; starting a new optimizer and schedule' % args.init_from)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.weight_decay)
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs)
    if args.warmup_epochs:
        warmup = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0 / args.warmup_epochs,
            total_iters=args.warmup_epochs)
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs])
    else:
        scheduler = cosine
    best_acc, start_epoch = 0.0, 0
    if checkpoint is not None:
        load_weights(net, checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        best_acc, start_epoch = checkpoint['best_acc'], checkpoint['epoch'] + 1
        print('==> Resuming at epoch %d with saved settings' % (start_epoch + 1))

    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: %d/%d | lr: %.6f' % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
        train(net, trainloader, criterion, optimizer, device, args.mixup_alpha)
        _, acc = test(net, testloader, criterion, device)
        scheduler.step()
        improved = acc > best_acc
        best_acc = max(best_acc, acc)
        model = net.module if isinstance(net, nn.DataParallel) else net
        state = {
            'net': model.state_dict(), 'acc': acc, 'best_acc': best_acc, 'epoch': epoch,
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'args': vars(args),
        }
        # Replace completed files so an interrupted write leaves the previous checkpoint intact.
        torch.save(state, last_path + '.tmp')
        os.replace(last_path + '.tmp', last_path)
        if improved:
            torch.save(state, best_path + '.tmp')
            os.replace(best_path + '.tmp', best_path)
            print('==> Saved best checkpoint: %.2f%%' % best_acc)


if __name__ == '__main__':
    main()
