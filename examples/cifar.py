'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import os, argparse, shutil, sys
from time import time

sys.path.append(os.path.dirname(__file__) + '/..')  # import from parent directory
from curveball import CurveBall

import models

try:
  from overboard import Logger
except ImportError:
  print('Warning: OverBoard not installed, no logging/plotting will be performed. See https://pypi.org/project/overboard/')
  Logger = None


def train(args, net, device, train_loader, optimizer, epoch, logger):
  net.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    start = time()
    data, target = data.to(device), target.to(device)

    # create closures to compute the forward pass, and the loss
    model_fn = lambda: net(data)
    loss_fn = lambda pred: F.cross_entropy(pred, target)

    if isinstance(optimizer, CurveBall):
      (loss, predictions) = optimizer.step(model_fn, loss_fn)
    else:
      # standard optimizer
      optimizer.zero_grad()
      predictions = model_fn()
      loss = loss_fn(predictions)
      loss.backward()
      optimizer.step()
    
    pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
    accuracy = pred.eq(target.view_as(pred)).double().mean()
    
    # log the loss and accuracy
    stats = {'train.loss': loss.item(), 'train.accuracy': accuracy.item()}
    if logger:
      logger.update_average(stats)
      if logger.avg_count['train.loss'] > 3:  # skip first 3 iterations (warm-up time)
        logger.update_average({'train.time': time() - start})
      logger.print(line_prefix='ep %i ' % epoch, prefix='train')
    else:
      print(stats)


def test(args, net, device, test_loader, logger):
  net.eval()
  with torch.no_grad():
    for data, target in test_loader:
      start = time()
      data, target = data.to(device), target.to(device)
      predictions = net(data)
      
      loss = F.cross_entropy(predictions, target)
      
      pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
      accuracy = pred.eq(target.view_as(pred)).double().mean()

      # log the loss and accuracy
      stats = {'val.loss': loss.item(), 'val.accuracy': accuracy.item()}
      if logger:
        logger.update_average(stats)
        if logger.avg_count['val.loss'] > 3:  # skip first 3 iterations (warm-up time)
          logger.update_average({'val.time': time() - start})
        logger.print(prefix='val')
      else:
        print(stats)


def main():
  all_models = [name for name in dir(models) if callable(getattr(models, name))]

  parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
  parser.add_argument("experiment", nargs='?', default="test")
  parser.add_argument('-model', choices=all_models, default='BasicNetBN')  #ResNet18
  parser.add_argument('-optimizer', choices=['sgd', 'adam', 'curveball'], default='curveball')  
  parser.add_argument('-lr', default=-1, type=float, help='learning rate')
  parser.add_argument('-momentum', type=float, default=-1, metavar='M')
  parser.add_argument('-lambda', type=float, default=1.0)
  parser.add_argument('--no-auto-lambda', action='store_true', default=False, help='disables automatic lambda estimation')
  parser.add_argument('-batch-size', default=128, type=int)
  parser.add_argument('-epochs', default=200, type=int)
  parser.add_argument('-save-interval', default=10, type=int)
  parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
  parser.add_argument('-outputdir', default='data/cifar-experiments', type=str)
  parser.add_argument('-datadir', default='data/cifar', type=str)
  parser.add_argument('-device', default='cuda', type=str)
  parser.add_argument('--parallel', action='store_true', default=False)
  args = parser.parse_args()

  args.outputdir += ('/' + args.model + '/' + args.optimizer + '/' + args.experiment)

  if os.path.isdir(args.outputdir):
    input('Directory already exists. Press Enter to overwrite or Ctrl+C to cancel.')

  if not torch.cuda.is_available(): args.device = 'cpu'
  best_acc = 0  # best test accuracy
  start_epoch = 0  # start from epoch 0 or last checkpoint epoch

  # data
  transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=2, fill=(128, 128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])

  transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])

  train_set = torchvision.datasets.CIFAR10(root=args.datadir, train=True, download=True, transform=transform_train)

  train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=2, shuffle=True)

  test_set = torchvision.datasets.CIFAR10(root=args.datadir, train=False, download=True, transform=transform_test)

  test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, num_workers=2, shuffle=False)

  # model
  net = getattr(models, args.model)()

  net = net.to(args.device)
  if args.device != 'cpu' and args.parallel:
    net = torch.nn.DataParallel(net)
  torch.backends.cudnn.benchmark = True  # slightly faster for fixed batch/input sizes

  if args.resume:
    # load checkpoint
    print('Resuming from checkpoint..')
    assert os.path.isdir(args.outputdir), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.outputdir + '/last.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
  
  # optimizer
  if args.optimizer == 'sgd':
    if args.lr < 0: args.lr = 0.1
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    
  elif args.optimizer == 'adam':
    if args.lr < 0: args.lr = 0.001
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

  elif args.optimizer == 'curveball':
    #if args.lr < 0: args.lr = 0.01
    #if args.momentum < 0: args.momentum = 0.9
    lambd = getattr(args, 'lambda')

    optimizer = CurveBall(net.parameters(), lr=args.lr, momentum=args.momentum, lambd=lambd, auto_lambda=not args.no_auto_lambda)

  logger = None
  if Logger: logger = Logger(args.outputdir, meta=args, resume=args.resume)

  for epoch in range(start_epoch, args.epochs):
    train(args, net, args.device, train_loader, optimizer, epoch, logger)
    test(args, net, args.device, test_loader, logger)
    
    if logger:
      acc = logger.average()['val.accuracy']
      logger.append()
    
    # save checkpoint
    if epoch % args.save_interval == 0:
      print('Saving..')
      state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'acc': acc, 'epoch': epoch}
      if not os.path.isdir(args.outputdir):
        os.mkdir(args.outputdir)
      torch.save(state, args.outputdir + '/last.t7')
      if logger and acc > best_acc:
        shutil.copyfile(args.outputdir + '/last.t7', args.outputdir + '/best.t7')
        best_acc = acc

if __name__ == '__main__':
  main()
