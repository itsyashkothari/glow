import numpy as np
import torch.nn as nn
import torch.nn.utils as utils
import torch
from torch.distributions import MultivariateNormal
from deepgenmodels.utils import MixtureDistribution

def bits_per_dim(x, nll):
    """Get the bits per dimension implied by using model with `loss`
    for compressing `x`, assuming each entry can take on `k` discrete values.

    Args:
        x (torch.Tensor): Input to the model. Just used for dimensions.
        nll (torch.Tensor): Scalar negative log-likelihood loss tensor.

    Returns:
        bpd (torch.Tensor): Bits per dimension implied if compressing `x`.
    """
    dim = np.prod(x.size()[1:])
    bpd = nll / (np.log(2) * dim)

    return bpd


def clip_grad_norm(optimizer, max_norm, norm_type=2):
    """Clip the norm of the gradients for all parameters under `optimizer`.

    Args:
        optimizer (torch.optim.Optimizer):
        max_norm (float): The maximum allowable norm of gradients.
        norm_type (int): The type of norm to use in computing gradient norms.
    """
    for group in optimizer.param_groups:
        utils.clip_grad_norm_(group['params'], max_norm, norm_type)


class NLLLoss(nn.Module):
    """Negative log-likelihood loss assuming isotropic gaussian with unit norm.

    Args:
        k (int or float): Number of discrete values in each input dimension.
            E.g., `k` is 256 for natural images.

    See Also:
        Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """
    def __init__(self, k=256):
        super(NLLLoss, self).__init__()
        self.k = k

    def forward(self, z, sldj):
        
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = prior_ll.flatten(1).sum(-1) \
            - np.log(self.k) * np.prod(z.size()[1:])
        print(prior_ll.shape)
        ll = prior_ll + sldj
        
        nll = -ll.mean()

        return nll

class NLLLoss2(nn.Module):
    """Negative log-likelihood loss assuming isotropic gaussian with unit norm.

    Args:
        k (int or float): Number of discrete values in each input dimension.
            E.g., `k` is 256 for natural images.

    See Also:
        Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """
    def __init__(self, inp_dims,num_classes,k=256):
        super(NLLLoss2, self).__init__()

        locs = np.linspace(-1 * np.ones(inp_dims), 1 * np.ones(inp_dims), num=num_classes)
        self.base_dist = MixtureDistribution(dists=[
            MultivariateNormal(loc=torch.as_tensor(loc).to('cuda'), covariance_matrix=torch.eye(inp_dims).type(torch.DoubleTensor).to('cuda')/2) for loc in locs
        ], dims=2)
        self.k = k

    def forward(self, z, sldj,y):
        z_hat = z.flatten(1)
        # print(y.shape)
        one_hot = torch.eye(10)[y].to('cuda')
        # print(one_hot)
        # print("yo",z_hat.shape)
        log_pzs_classwise = self.base_dist.log_probs_classwise(z_hat)
        prior_ll = torch.sum(log_pzs_classwise*one_hot,dim=1)
        # print(prior_ll.shape)
        # prior_ll = log_pzs_classwise.mean(dim=1)
        # print(prior_ll)
        # print(log_pzs_classwise)
        # log_pzs_min = torch.min(log_pzs_classwise, dim=1, keepdim=True)[0]
        # log_pzs_max = torch.max(log_pzs_classwise, dim=1, keepdim=True)[0]
        # print(log_pzs_min,log_pzs_max,log_pzs_max-log_pzs_min)

        # log_pzs_classwise -= log_pzs_min
        # pzs_classwise = torch.exp(log_pzs_classwise)
        # print(pzs_classwise)
        # log_pzs_min = log_pzs_min.reshape(-1)
        # log_pz = log_pzs_min + torch.log(torch.sum(pzs_classwise,dim=1))
        # log_pz = torch.log(torch.sum(pzs_classwise,dim=1))
        # print(torch.min(log_pz))
        # log_pz = log_pzs_min + torch.log(torch.sum(torch.mul(pzs_classwise, class_probs), dim=1))
        # prior_ll = log_pz
        ll = prior_ll + sldj
        nll = -ll.mean()
        # print(nll)
        return nll
