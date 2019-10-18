
import torch.nn as nn


class Flatten(nn.Module):
  def forward(self, input):
    return input.view(input.size(0), -1)

def BasicNetBN():
  return BasicNet(batch_norm=True)

def BasicNet(batch_norm=False):
  """Basic network for CIFAR."""
  layers = [
    nn.Conv2d(3, 32, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

    nn.Conv2d(32, 32, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

    nn.Conv2d(32, 64, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
    
    Flatten(),

    nn.Linear(4 * 4 * 64, 64),
    nn.ReLU(),

    nn.Linear(64, 10)
  ]

  # insert batch norm layers
  if batch_norm:
    insert_bnorm(layers, init_gain=True, eps=1e-4)

  return nn.Sequential(*layers)


def insert_bnorm(layers, init_gain=False, eps=1e-5, ignore_last_layer=True):
  """Inserts batch-norm layers after each convolution/linear layer in a list of layers."""
  last = True
  for (idx, layer) in reversed(list(enumerate(layers))):
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
      if ignore_last_layer and last:
        last = False  # do not insert batch-norm after last linear/conv layer
      else:
        if isinstance(layer, nn.Conv2d):
          bnorm = nn.BatchNorm2d(layer.out_channels, eps=eps)
        elif isinstance(layer, nn.Linear):
          bnorm = nn.BatchNorm1d(layer.out_features, eps=eps)
        
        if init_gain:
          bnorm.weight.data[:] = 1.0  # instead of uniform sampling

        layers.insert(idx + 1, bnorm)
  return layers
