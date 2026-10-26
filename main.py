'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
from datetime import datetime

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

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--arch', default='ResNet18', choices=sorted(model_factories),
                    help='model architecture')
parser.add_argument('--run-name', help='unique experiment name; required with --resume')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()
if args.resume and not args.run_name:
    parser.error('--resume requires --run-name to select an experiment')
if args.run_name and (args.run_name in ('.', '..') or
                      os.path.basename(args.run_name) != args.run_name):
    parser.error('--run-name must be a single directory name')
run_name = args.run_name or datetime.now().strftime('%Y%m%d-%H%M%S-%f')
checkpoint_dir = os.path.join('checkpoint', args.arch, run_name)
checkpoint_path = os.path.join(checkpoint_dir, 'ckpt.pth')
if not args.resume:
    os.makedirs(checkpoint_dir, exist_ok=False)
print('==> Model: %s | Run: %s | Checkpoint: %s' %
      (args.arch, run_name, checkpoint_path))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
data_root = '/home/wzc/nas/home/Dataset/CIFAR10/CIFAR10'
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.ImageFolder(
    root=os.path.join(data_root, 'train'), transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=256, shuffle=True, num_workers=2)

testset = torchvision.datasets.ImageFolder(
    root=os.path.join(data_root, 'test'), transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = trainset.classes

# Model
print('==> Building model..')
net = model_factories[args.arch]()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(checkpoint_path), 'Error: no checkpoint found: %s' % checkpoint_path
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, checkpoint_path)
        best_acc = acc


for epoch in range(start_epoch, start_epoch+300):
    train(epoch)
    test(epoch)
    scheduler.step()
