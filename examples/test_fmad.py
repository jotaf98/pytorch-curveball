
import torch as t
from torch.autograd import grad

def fmad(ys, xs, dxs):
  # inspired by: https://github.com/renmengye/tensorflow-forward-ad/issues/2
  v = [t.zeros_like(y, requires_grad=True) for y in ys]
  g = grad(ys, xs, grad_outputs=v, create_graph=True)
  return grad(g, v, grad_outputs=dxs)


# linear function test
x = t.tensor([[0.1], [0.2]], requires_grad=True)
A = t.tensor([[1, 2], [3, 4], [5, 6]], dtype=t.float)
y = A @ x

print('Linear function output:\n', y)

bwd_der = grad([y], [x], [t.ones_like(y)], retain_graph=True)
print('RMAD:\n', bwd_der)

print('Closed-form backward gradient:\n', A.t() @ t.ones_like(y))

fwd_der = fmad([y], [x], [t.ones_like(x)])
print('FMAD:\n', fwd_der)

print('Closed-form forward gradient:\n', A @ t.ones_like(x))


# MLP test
net = t.nn.Sequential(
  t.nn.Linear(3, 4),
  t.nn.ReLU(),
  t.nn.Linear(4, 2)
)
parameters = list(net.parameters())
output = net(t.ones((1, 3)))

fwd_der = fmad([output], parameters, [t.ones_like(p) for p in parameters])
print('MLP FMAD:\n', fwd_der)

# numerical check
loss_der = t.randn_like(output)  # arbitrary
param_der = [t.zeros_like(p) for p in parameters]
result_der = [t.zeros_like(p) for p in parameters]

for (param_idx, param) in enumerate(parameters):
  for elem in range(param.numel()):
    # set one-hot tensor
    param_der[param_idx].flatten()[elem] = 1

    fmad_result = fmad([output], parameters, param_der)
    assert len(fmad_result) == 1

    result_der[param_idx].flatten()[elem] = (fmad_result[0] * loss_der).sum()

    param_der[param_idx].flatten()[elem] = 0  # reset it

print('MLP backward gradient using FMAD (numerical):', result_der)

print('MLP backward gradient using RMAD (standard):', grad([output], parameters, loss_der))

