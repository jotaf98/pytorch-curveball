
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
    last = True
    for (idx, layer) in reversed(list(enumerate(layers))):
      if last and isinstance(layer, (nn.Conv2d, nn.Linear)):
        last = False  # do not insert batch-norm after last linear/conv layer
      else:
        if isinstance(layer, nn.Conv2d):
          layers.insert(idx + 1, nn.BatchNorm2d(layer.out_channels))
        elif isinstance(layer, nn.Linear):
          layers.insert(idx + 1, nn.BatchNorm1d(layer.out_features))

  return nn.Sequential(*layers)
