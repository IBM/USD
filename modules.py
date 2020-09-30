import torch
import torch.nn as nn
from torch.autograd import grad
from torch.distributions.categorical import Categorical
import numpy as np
from scipy.spatial.distance import cdist


class RFFEmbedding(nn.Module):
    r"""Random Fourier Features Embedding

    Args
        **num_features** (scalar): number of input features
        **num_outputs** (scalar): number of random Fourier features
        **sigma** (scalar): kernel bandwidth

    Inputs
        **inputs** (batch x num_features): batch of inputs

    Outputs
        **outputs** (batch x num_outputs): batch of embedded inputs
    """
    def __init__(self, num_features, num_outputs=100, sigma=1.0):
        super(RFFEmbedding, self).__init__()

        self.num_features = num_features
        self.num_outputs = num_outputs
        self.sigma = sigma

        self.weight = nn.Parameter(torch.Tensor(num_features, num_outputs).normal_().mul_(np.sqrt(2) / sigma))
        self.bias = nn.Parameter(torch.Tensor(num_outputs).uniform_(-np.pi, np.pi))

    def forward(self, inputs):
        h = inputs @ self.weight + self.bias
        return torch.cos(h).mul(np.sqrt(2 / self.num_outputs))


class MMD_RFF(nn.Module):
    r"""MMD computed with Random Fourier Features

    Args
        **num_features** (scalar): number of input features
        **num_outputs** (scalar): number of random Fourier features

    Inputs
        **X** (batch1 x num_features): batch of inputs from distribution X
        **Y** (batch2 x num_features): batch of inputs from distribution Y
        **weights_X** (batch1, optional): weights weighing samples from X
            Weights are normalized so that weights_X.sum() == 1
        **weights_Y** (batch2, optional): weights weighing samples from Y
            Weights are normalized so that weights_X.sum() == 1

    Outputs
        **mmd**: Maximum Mean Discrepancy between X and Y
    """
    def __init__(self, num_features, num_outputs=100, sigma=1.0):
        super(MMD_RFF, self).__init__()

        self.num_features = num_features
        self.num_outputs = num_outputs

        self.rff_emb = RFFEmbedding(num_features, num_outputs, sigma=sigma)

    def forward(self, X, Y, weights_X=None, weights_Y=None):
        fX, fY = self.rff_emb(X), self.rff_emb(Y)

        if weights_X is None:
            mu_X = fX.mean(0)
        else:

            mu_X = (weights_X.view(-1,1) / weights_X.sum() * fX).sum(0)

        if weights_Y is None:
            mu_Y = fY.mean(0)
        else:
            mu_Y = (weights_Y.view(-1,1) / weights_Y.sum() * fY).sum(0)

        d_XY = mu_X - mu_Y
        return d_XY.norm()


def sparsemax(logits):
    r"""SparseMax (only forward step)

    Inputs
        **logits** (batch x num_features): input logits
    """
    assert logits.dim() == 2, "This module only works with 2D tensors (batch x num_features)"
    n_logits = logits.shape[-1]
    assert n_logits > 1, "There should be more features. Check that the inputs tensor is batch x num_features"
    device = logits.device

    # Translate inputs by max for numerical stability
    logits = logits - torch.max(logits, dim=-1, keepdim=True)[0].expand_as(logits)

    # Sort input in descending order.
    zs = torch.sort(logits, dim=-1, descending=True)[0]
    rng = torch.arange(1.0, n_logits + 1.0, device=device).expand_as(logits)

    # Determine sparsity of projection
    cumsum = torch.cumsum(zs, -1)
    is_gt = torch.gt(1 + rng * zs, cumsum).type(logits.type())
    k = torch.max(is_gt * rng, -1, keepdim=True)[0]

    # Compute taus
    taus = (torch.sum(is_gt * zs, -1, keepdim=True) - 1) / k
    taus = taus.expand_as(logits)

    outputs = torch.max(torch.zeros_like(logits), logits - taus)
    return outputs


class LpNorm(nn.Module):
    '''Lp-Normalization
        Normalizes inputs by dividing by norm(inputs, p)
    '''
    def __init__(self, p=2, eps=1e-6):
        super(LpNorm, self).__init__()
        self.eps = eps
        self.p = p

    def forward(self, inputs):
        norm = inputs.norm(self.p, -1, keepdim=True).add(self.eps)
        return inputs.div(norm)

    def extra_repr(self):
        return '{p}'.format(**self.__dict__)


class D_mlp(nn.Module):
    r"""MLP discriminator
    """
    def __init__(self, insize=2, layerSizes=[32,32,16], nonlin='ReLU', normalization=None, dropout=False):
        super(D_mlp, self).__init__()
        self.phi = nn.Sequential()
        for ix, inSi, outSi in zip(range(len(layerSizes)), [insize]+layerSizes[:-1], layerSizes):
            self.phi.add_module('L'+str(ix), nn.Linear(inSi, outSi))
            if normalization and ix == 1:
                if normalization[0] =='L':
                    self.phi.add_module('A'+str(ix), nn.LayerNorm(outSi))
                if normalization[0] == 'B':
                    self.phi.add_module('A'+str(ix), nn.BatchNorm1d(outSi, track_running_stats=False, momentum=0.0))
                if normalization[0] == 'N':
                    self.phi.add_module('A'+str(ix), LpNorm())
            if nonlin == 'LeakyReLU':
                self.phi.add_module('N'+str(ix), nn.LeakyReLU(0.2, inplace=True))
            elif nonlin == 'ReLU':
                self.phi.add_module('N'+str(ix), nn.ReLU(inplace=True))
            elif nonlin == 'Sigmoid':
                self.phi.add_module('N'+str(ix), nn.Sigmoid())
            if ix == 1 and dropout: # droupout only after first layer
                self.phi.add_module('D'+str(ix), nn.Dropout(dropout))

        self.V = nn.Linear(layerSizes[-1], 1, bias=False)

    def forward(self, input):
        x = self.phi(input)
        return self.V(x)


class D_mlp_norm(nn.Module):
    r"""MLP discriminator
    """
    def __init__(self, insize=2, layerSizes=[32,32,16], nonlin='ReLU', normalization=None):
        super(D_mlp_norm, self).__init__()
        self.phi = nn.Sequential()
        for ix, inSi, outSi in zip(range(len(layerSizes)), [insize]+layerSizes[:-1], layerSizes):
            self.phi.add_module('L'+str(ix), nn.Linear(inSi, outSi))
            if ix > 0 and normalization: # only LN/IN after first layer.
                if normalization == 'LN':
                    self.phi.add_module('A'+str(ix), nn.LayerNorm(outSi))
            if nonlin == 'LeakyReLU':
                self.phi.add_module('N'+str(ix), nn.LeakyReLU(0.2, inplace=True))
            elif nonlin == 'ReLU':
                self.phi.add_module('N'+str(ix), nn.ReLU(inplace=True))
            elif nonlin == 'Sigmoid':
                self.phi.add_module('N'+str(ix), nn.Sigmoid())
        self.V = nn.Linear(layerSizes[-1], 1, bias=False)

    def forward(self, input):
        x = self.phi(input)
        return self.V(x / (x.norm(dim=-1, keepdim=True) + 1e-6))


def D_forward_weights(D, x_p, w_p, x_q, w_q, lambda_aug, alpha, rho):
    """Computes the objective but returns the loss = -obj
    """
    x_q.requires_grad_(True)
    if w_p is None:
        w_p = 1.0

    f_p, f_q = D(x_p), D(x_q)
    Ep_f = (w_p * f_p).mean()
    Eq_f = (w_q * f_q).mean()

    # FISHER
    constraint_F = (w_q * f_q**2).mean() - Eq_f**2

    # SOBOLEV
    grad_f_q = grad(outputs=Eq_f, inputs=x_q, create_graph=True)[0]
    normgrad_f2_q = (grad_f_q**2).sum(dim=1, keepdim=True)
    constraint_S = (w_q * normgrad_f2_q).mean()

    # Combining FISHER and SOBOLEV constraints
    constraint_tot = (constraint_S + alpha * constraint_F - 1.0)

    obj_D = Ep_f - Eq_f \
            - lambda_aug * constraint_tot - rho/2  * constraint_tot**2

    return -obj_D, Ep_f, Eq_f, normgrad_f2_q


class KMeansPlusPlus(object):
    r"""This version of kmeans++ initialization accepts weights to weight the samples
    """
    def __init__(self, data, weights=None):
        self.n_data = data.shape[0]
        self.n_features = data.shape[-1]

        self.data = data
        self.weights = weights

        self.mu = np.zeros((0, self.n_features)) # centroids
        self.mu_idx = []  # index of centroids in self.data
        self.D2 = np.full((1, self.n_data), np.inf)

    def _add_centroid(self, center):
        if center.ndim == 1:
            center = center.reshape(1, -1)
        self.mu = np.r_[self.mu, center]

        # Recompute distances to closest centroid
        D2_add = cdist(center, self.data, 'sqeuclidean')
        self.D2 = np.min(np.r_[self.D2, D2_add], axis=0, keepdims=True)

    def _choose_uniformly(self):
        if self.weights is None:
            probs = np.ones((1, self.n_data)) / self.n_data
        else:
            probs = self.weights / self.weights.sum()

        distr = Categorical(torch.tensor(probs))
        idx = distr.sample().numpy()
        self.mu_idx.append(idx.item())
        return self.data[idx].reshape(1, -1)

    def _choose_next_centroid(self):
        probs = self.D2 / self.D2.sum()
        if self.weights is not None:
            probs_w = self.weights / self.weights.sum()
            probs = probs * probs_w
            probs = probs / probs.sum()

        distr = Categorical(torch.tensor(probs))
        idx = distr.sample().numpy()
        self.mu_idx.append(idx.item())
        return self.data[idx].reshape(1, -1)

    def init_centroids(self, k):
        # Sample first centroid uniformly
        c = self._choose_uniformly()
        self._add_centroid(c)

        # Sample the rest using kmeans++ algorithm
        for _ in range(k - 1):
            c = self._choose_next_centroid()
            self._add_centroid(c)


def swish(inputs):
    return inputs * torch.sigmoid(inputs)


def dswish(inputs):
    outputs = swish(inputs) + torch.sigmoid(inputs) * (1 - swish(inputs))
    return outputs


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inputs):
        return swish(inputs)


class DSwish(nn.Module):

    def __init__(self):
        super(DSwish, self).__init__()

    def forward(self, inputs):
        return dswish(inputs)


class IntegrableMLP(nn.Module):
    r"""Integrable MLP
    `Sequential` module `phi` executes the integrated architecture, up until the last layer
    `int_forward()` returns V.t() @ phi
    `forward()` executes the integrable architecture V.t() @ df/dx

    Outputs: output
        - **output'**: (batch, num_inputs, num_outputs): tensor containing the Jacobian df_i/dx_j
    """
    def __init__(self, n_inputs=2, layers=[32,32,16]):
        super(IntegrableMLP, self).__init__()

        self.n_inputs = n_inputs
        self.n_outputs = layers[-1]
        self.layers = layers

        self.phi = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip([n_inputs]+layers[:-1], layers)):
            self.phi.add_module('L'+str(i), nn.Linear(in_size, out_size))
            self.phi.add_module('N'+str(i), Swish())

        self.V = nn.Linear(layers[-1], 1, bias=False)

    def forward(self, inputs):
        batch = inputs.shape[0]

        a = inputs
        alpha = torch.eye(self.n_inputs).repeat(batch,1,1)
        for l, _ in enumerate(self.layers):
            W = getattr(self.phi, 'L'+str(l))
            f = getattr(self.phi, 'N'+str(l))

            z = W(a)
            a = f(z)
            beta = alpha @ W.weight.t()
            alpha = dswish(z).unsqueeze(1).repeat(1,self.n_inputs,1) * beta

        return self.V(alpha).view(batch, self.n_inputs)

    def int_forward(self, inputs):
        x = self.phi(inputs)
        return self.V(x)


def manual_sgd_(x, lr):
    # artisanal sgd. Note we minimze alpha so a <- a + lr * grad
    x.data += lr * x.grad
    x.grad.zero_()
