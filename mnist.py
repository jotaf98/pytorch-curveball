
# Modified version of the PyTorch MNIST example to log outputs for OverBoard

from __future__ import print_function
import argparse, sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from time import time

from curveball import CurveBall

from overboard import Logger

class Flatten(nn.Module):
  def forward(self, input):
    return input.view(input.size(0), -1)

def onehot(target, like):
  """Transforms numeric labels into one-hot regression targets."""
  out = torch.zeros_like(like)
  out.scatter_(1, target.unsqueeze(1), 1.0)
  return out

def train(args, model, device, train_loader, optimizer, epoch, logger):
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    start = time()
    data, target = data.to(device), target.to(device)

    if args.loss == 'crossentropy':
      # create closures to compute the forward pass, and the loss
      model_fn = lambda: model(data)
      loss_fn = lambda pred: F.cross_entropy(pred, target)
    else:
      # MSE requires converting numeric labels to one-hot regression targets
      model_fn = lambda: F.softmax(model(data), dim=1)
      loss_fn = lambda pred: F.mse_loss(pred, onehot(target, pred))

    if isinstance(optimizer, CurveBall):
      (loss, predictions) = optimizer.step(model_fn, loss_fn)
    else:
      predictions = model_fn()
      loss = loss_fn(predictions)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
    
    pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
    accuracy = pred.eq(target.view_as(pred)).double().mean()
    
    # log the loss and accuracy
    logger.update_average({'train.loss': loss.item(), 'train.accuracy': accuracy.item()})

    if logger.avg_count['train.loss'] > 3:  # skip first 3 iterations (warm-up time)
      logger.update_average({'train.time': time() - start})

    logger.print(prefix='train')

def test(args, model, device, test_loader, logger):
  model.eval()
  with torch.no_grad():
    for data, target in test_loader:
      start = time()
      data, target = data.to(device), target.to(device)
      predictions = model(data)
        
      if args.loss == 'crossentropy':
        loss = F.cross_entropy(predictions, target)
      else:
        predictions = F.softmax(predictions, dim=1)
        loss = F.mse_loss(predictions, onehot(target, predictions))

      pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
      accuracy = pred.eq(target.view_as(pred)).double().mean()

      # log the loss and accuracy
      logger.update_average({'val.loss': loss.item(), 'val.accuracy': accuracy.item()})

      if logger.avg_count['val.loss'] > 3:  # skip first 3 iterations (warm-up time)
        logger.update_average({'val.time': time() - start})

  # display final values in console
  logger.print(prefix='val')

def main():
  # Training settings
  parser = argparse.ArgumentParser()
  parser.add_argument("experiment", nargs='?', default="")
  parser.add_argument('--batch-size', type=int, default=64, metavar='N',
            help='input batch size for training (default: 64)')
  parser.add_argument('--test-batch-size', type=int, default=1000,
            help='input batch size for testing (default: 1000)')
  parser.add_argument('--epochs', type=int, default=10,
            help='number of epochs to train (default: 10)')
  parser.add_argument('--loss', choices=['crossentropy', 'mse'], default='crossentropy',
            help='loss function')
  parser.add_argument('--optimizer', choices=['sgd', 'adam', 'curveball'], default='curveball',
            help='optimizer (sgd, adam, or curveball)')
  parser.add_argument('--lr', type=float, default=-1, metavar='LR',
            help='learning rate (default: 0.01 for SGD, 0.001 for Adam, 1 for CurveBall)')
  parser.add_argument('--momentum', type=float, default=-1, metavar='M',
            help='momentum (default: 0.5)')
  parser.add_argument('--lambda', type=float, default=1.0,
            help='lambda')
  parser.add_argument('--no-auto-lambda', action='store_true', default=False,
            help='disables automatic lambda estimation')
  parser.add_argument('--no-batch-norm', action='store_true', default=False)
  parser.add_argument('--no-cuda', action='store_true', default=False,
            help='disables CUDA training')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
            help='random seed (default: 1)')
  parser.add_argument('--datadir', type=str, default='C:\\data\\mnist\\',
            help='MNIST data directory')
  parser.add_argument('--outputdir', type=str, default='C:\\data\\mnist-experiments\\',
            help='output directory')
  args = parser.parse_args()
  use_cuda = not args.no_cuda and torch.cuda.is_available()
  args.outputdir += '/' + args.loss + '/' + args.optimizer + '/' + args.experiment

  if os.path.isdir(args.outputdir):
    input('Directory already exists. Press Enter to overwrite or Ctrl+C to cancel.')

  torch.manual_seed(args.seed)

  device = torch.device("cuda" if use_cuda else "cpu")

  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
  train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(args.datadir, train=True, download=True,
             transform=transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.1307,), (0.3081,))
             ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
  test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(args.datadir, train=False, transform=transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.1307,), (0.3081,))
             ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

  # same network as in the tutorial, in sequential form, and with optional batch-norm
  layers = [
    nn.Conv2d(1, 10, kernel_size=5),
    nn.MaxPool2d(kernel_size=2),
    nn.ReLU(),
    nn.Conv2d(10, 20, kernel_size=5),
    nn.MaxPool2d(kernel_size=2),
    nn.ReLU(),
    nn.Dropout2d(),
    Flatten(),
    nn.Linear(320, 50),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(50, 10)
  ]

  # insert batch norm layers
  if not args.no_batch_norm:
    for (idx, layer) in enumerate(layers[:]):
      if isinstance(layer, nn.Conv2d):
        layers.insert(idx + 1, nn.BatchNorm2d(layer.out_channels))
      elif isinstance(layer, nn.Linear):
        layers.insert(idx + 1, nn.BatchNorm1d(layer.out_features))

  model = nn.Sequential(*layers)
  model.to(device)

  if args.optimizer == 'sgd':
    if args.lr < 0: args.lr = 0.01
    if args.momentum < 0: args.momentum = 0.5
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

  elif args.optimizer == 'adam':
    if args.lr < 0: args.lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

  elif args.optimizer == 'curveball':
    #if args.lr < 0: args.lr = 0.01
    #if args.momentum < 0: args.momentum = 0.9
    lambd = getattr(args, 'lambda')

    optimizer = CurveBall(model.parameters(), lr=args.lr, momentum=args.momentum, lambd=lambd, auto_lambda=not args.no_auto_lambda)

  # open logging stream
  with Logger(args.outputdir, meta=args) as logger:
    # do training
    for epoch in range(1, args.epochs + 1):
      train(args, model, device, train_loader, optimizer, epoch, logger)
      test(args, model, device, test_loader, logger)

      # record average statistics collected over this epoch (with logger.update_average)
      logger.append()


if __name__ == '__main__':
  main()

