
import torch as t
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.autograd import grad


class CurveBall(Optimizer):
  """CurveBall optimizer"""
  def __init__(self, params, lr=None, momentum=None, auto_lambda=True, lambd=10.0,
      lambda_factor=0.999, lambda_low=0.5, lambda_high=1.5, lambda_interval=5):
    
    defaults = dict(lr=lr, momentum=momentum, auto_lambda=auto_lambda,
      lambd=lambd, lambda_factor=lambda_factor, lambda_low=lambda_low,
      lambda_high=lambda_high, lambda_interval=lambda_interval)
    super().__init__(params, defaults)

  def step(self, model_fn, loss_fn):
    """Performs a single optimization step"""

    # only support one parameter group
    if len(self.param_groups) != 1:
      raise ValueError('Since the hyper-parameters are set automatically, only one parameter group (with the same hyper-parameters) is supported.')
    group = self.param_groups[0]
    parameters = group['params']

    # initialize state to 0 if needed
    state = self.state
    for p in parameters:
      if p not in state:
        state[p] = {'z': t.zeros_like(p)}
    
    # linear list of state tensors z
    zs = [state[p]['z'] for p in parameters]
    
    # store global state (step count, lambda estimate) with first parameter
    global_state = state[parameters[0]]
    global_state.setdefault('count', 0)

    # get lambda estimate, or initial lambda (user hyper-parameter) if it's not set
    lambd = global_state.get('lambd', group['lambd'])
    

    #
    # compute CurveBall step (delta_zs)
    #

    # run forward pass, cutting off gradient propagation between model and loss function for efficiency
    predictions = model_fn()
    predictions_d = predictions.detach().requires_grad_(True)
    loss = loss_fn(predictions_d)

    # compute J^T * z using FMAD (where z are the state variables)
    (Jz,) = fmad(predictions, parameters, zs)  # equivalent but slower
    
    # compute loss gradient Jl, retaining the graph to allow 2nd-order gradients
    (Jl,) = grad(loss, predictions_d, create_graph=True)
    Jl_d = Jl.detach()  # detached version, without requiring gradients

    # compute loss Hessian (projected by Jz) using 2nd-order gradients
    (Hl_Jz,) = grad(Jl, predictions_d, grad_outputs=Jz, retain_graph=True)

    # compute J * (Hl_Jz + Jl) using RMAD (back-propagation).
    # note this is still missing the lambda * z term.
    delta_zs = grad(predictions, parameters, Hl_Jz + Jl_d, retain_graph=True)
    
    # add lambda * z term to the result, obtaining the final steps delta_zs
    for (z, dz) in zip(zs, delta_zs):
      dz.data.add_(lambd, z)


    #
    # automatic hyper-parameters: momentum (rho) and learning rate (beta)
    #

    lr = group['lr']
    momentum = group['momentum']

    if momentum < 0 or lr < 0 or group['auto_lambda']:  # required by auto-lambda
      # compute J^T * delta_zs
      (Jdeltaz,) = fmad(predictions, parameters, delta_zs)  # equivalent but slower

      # project result by loss hessian (using 2nd-order gradients)
      (Hl_Jdeltaz,) = grad(Jl, predictions_d, grad_outputs=Jdeltaz)

      # solve 2x2 linear system: [rho, -beta]^T = [a11, a12; a12, a22]^-1 [b1, b2]^T.
      # accumulate components of dot-product from all parameters, by first aggregating them into a vector.
      z_vec = t.cat([z.flatten() for z in zs])
      dz_vec = t.cat([dz.flatten() for dz in delta_zs])

      a11 = lambd * (dz_vec * dz_vec).sum() + (Jdeltaz * Hl_Jdeltaz).sum()
      a12 = lambd * (dz_vec * z_vec).sum() + (Jz * Hl_Jdeltaz).sum()
      a22 = lambd * (z_vec * z_vec).sum() + (Jz * Hl_Jz).sum()

      b1 = (Jl_d * Jdeltaz).sum()
      b2 = (Jl_d * Jz).sum()

      # item() implicitly moves to the CPU
      A = t.tensor([[a11.item(), a12.item()], [a12.item(), a22.item()]])
      b = t.tensor([[b1.item()], [b2.item()]])
      auto_params = A.pinverse() @ b

      lr = auto_params[0].item()
      momentum = -auto_params[1].item()


    #
    # update parameters and state in-place: z = momentum * z + lr * delta_z; p = p + z
    #

    for (p, z, dz) in zip(parameters, zs, delta_zs):
      z.data.mul_(momentum).add_(-lr, dz)  # update state
      p.data.add_(z)  # update parameter


    #
    # automatic lambda hyper-parameter (trust region adaptation)
    #

    if group['auto_lambda']:
      # only adapt once every few batches
      if global_state['count'] % group['lambda_interval'] == 0:
        with t.no_grad():
          # evaluate the loss with the updated parameters
          new_loss = loss_fn(model_fn())
          
          # objective function change predicted by quadratic fit
          quadratic_change = -0.5 * (auto_params * b).sum()

          # ratio between predicted and actual change
          ratio = (new_loss - loss) / quadratic_change

          # increase or decrease lambda based on ratio
          factor = group['lambda_factor'] ** group['lambda_interval']

          if ratio < group['lambda_low']: lambd /= factor
          if ratio > group['lambda_high']: lambd *= factor
          
          global_state['lambd'] = lambd
      global_state['count'] += 1

    return (loss, predictions)


def fmad(ys, xs, dxs):
  """Forward-mode automatic differentiation."""
  v = t.zeros_like(ys, requires_grad=True)
  g = grad(ys, xs, grad_outputs=v, create_graph=True)
  return grad(g, v, grad_outputs=dxs)

