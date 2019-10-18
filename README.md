# PyTorch CurveBall - A second-order optimizer for deep networks

This is a PyTorch re-implementation of the CurveBall algorithm, presented in:

> João F. Henriques, Sebastien Ehrhardt, Samuel Albanie, Andrea Vedaldi, "Small Steps and Giant Leaps: Minimal Newton Solvers for Deep Learning", ICCV 2019 ([arXiv](https://arxiv.org/abs/1805.08095))

It follows closely the [original](https://github.com/jotaf98/curveball) Matlab code, although it does not implement all the experiments in that paper.

### Warning:

Unfortunately, the PyTorch operations used for forward-mode automatic differentiation (FMAD) are somewhat slow (refer to [this issue](https://github.com/pytorch/pytorch/issues/22577)).

For this reason, it is not as fast as the original Matlab implementation or this [TensorFlow](https://github.com/hyenal/curveball-tf) port.

This is something that we hope to resolve quickly, either by the PyTorch bug getting fixed, or by implementing a CurveBall variant that is inherently faster despite the slow operations. Other suggested fixes are very welcome.


## Requirements

Although it may work with older versions, this has mainly been tested with:

- PyTorch 1.3
- Python 3.7

Plots are produced with [OverBoard](https://pypi.org/project/overboard/) (optional).


## Usage

The optimizer does not need any hyper-parameters:

```
from curveball import CurveBall
net = ...  # your network goes here
optimizer = CurveBall(net.parameters()
```

Inside the training loop, CurveBall needs to know what loss you're using (or losses, as a sum). Provide them as a closure (function), for example (given some `labels`):

`loss_fn = lambda pred: F.cross_entropy(pred, labels)`

Similarly, provide a closure for the forward operation of the model (given a model `net` and input `images`):

`model_fn = lambda: net(images)`

Now the optimizer step is just:

`(loss, predictions) = optimizer.step(model_fn, loss_fn)`

Note that, unlike gradient-based optimizers like SGD, there's no need to run the model forward, call `backward()`, zero-gradients, or any other operations -- the optimizer's step will do all those things (by calling the closures you defined), and update the network's parameters. You can define more complex loss functions or models by using full functions (`def f(): ...`) instead of lambda functions.


# Full example

See `cifar.py` or `mnist.py`.


# Author

[João F. Henriques](http://www.robots.ox.ac.uk/~joao/)

