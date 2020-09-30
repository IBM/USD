import os
import time
from random import randint
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


class ddict:
    """DotDictionary: dictionary whose items can be accesses with the dot operator

        E.g.
        >> args = DDICT(batch_size=128, epochs=10)
        >> print(args.batch_size)
    """
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __repr__(self):
        return str(self.__dict__)

    def __iter__(self):
        return self.__dict__.__iter__()

    def __len__(self):
        return len(self.__dict__)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    def update(self, other):
        if type(other) == ddict:
            self.__dict__.update(other.__dict__)
        if type(other) == dict:
            self.__dict__.update(other)
        return self


def get_devices(cuda_device="cuda:0", seed=1):
    """Gets cuda devices
    """
    device = torch.device(cuda_device)
    torch.manual_seed(seed)
    # Multi GPU?
    num_gpus = torch.cuda.device_count()
    if device.type != 'cpu':
        print('\033[93m' + 'Using CUDA,', num_gpus, 'GPUs\033[0m')
        torch.cuda.manual_seed(seed)
    return device, num_gpus


def make_data_parallel(module, expose_methods=None):
    """Wraps `nn.Module object` into `nn.DataParallel` and links methods whose name is listed in `expose_methods`
    """
    dp_module = nn.DataParallel(module)

    if expose_methods is None:
        if hasattr(module, 'expose_methods'):
            expose_methods = module.expose_methods

    if expose_methods is not None:
        for mt in expose_methods:
            setattr(dp_module, mt, getattr(dp_module.module, mt))
    return dp_module


class shelf(object):
    '''Shelf to save stuff to disk. Basically a DDICT which can save to disk.

    Example:
        SH = shelf(lr=[0.1, 0.2], n_hiddens=[100, 500, 1000], n_layers=2)
        SH._extend(['lr', 'n_hiddens'], [[0.3, 0.4], [2000]])
        # Save to file:
        SH._save('my_file', date=False)
        # Load shelf from file:
        new_dd = shelf()._load('my_file')
    '''
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __add__(self, other):
        if isinstance(other, type(self)):
            sum_dct = copy.copy(self.__dict__)
            for k, v in other.__dict__.items():
                if k not in sum_dct:
                    sum_dct[k] = v
                else:
                    if type(v) is list and type(sum_dct[k]) is list:
                        sum_dct[k] = sum_dct[k] + v
                    elif type(v) is not list and type(sum_dct[k]) is list:
                        sum_dct[k] = sum_dct[k] + [v]
                    elif type(v) is list and type(sum_dct[k]) is not list:
                        sum_dct[k] = [sum_dct[k]] + v
                    else:
                        sum_dct[k] = [sum_dct[k]] + [v]
            return shelf(**sum_dct)

        elif isinstance(other, dict):
            return self.__add__(shelf(**other))
        else:
            raise ValueError("shelf or dict is required")

    def __radd__(self, other):
        return self.__add__(other)

    def __repr__(self):
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in self._keys())
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __iter__(self):
        return self.__dict__.__iter__()

    def __len__(self):
        return len(self.__dict__)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    @staticmethod
    def _flatten_dict(d, parent_key='', sep='_'):
        "Recursively flattens nested dicts"
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, MutableMapping):
                items.extend(shelf._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _extend(self, keys, values_list):
        if type(keys) not in (tuple, list):  # Individual key
            if keys not in self._keys():
                self[keys] = values_list
            else:
                self[keys] += values_list
        else:
            for key, val in zip(keys, values_list):
                if type(val) is list:
                    self._extend(key, val)
                else:
                    self._extend(key, [val])
        return self

    def _keys(self):
        return tuple(sorted([k for k in self.__dict__ if not k.startswith('_')]))

    def _values(self):
        return tuple([self.__dict__[k] for k in self._keys()])

    def _items(self):
        return tuple(zip(self._keys(), self._values()))

    def _save(self, filename=None, date=True):
        if filename is None:
            if not hasattr(self, '_filename'):  # First save
                raise ValueError("filename must be provided the first time you call _save()")
            else:  # Already saved
                torch.save(self, self._filename + '.pt')
        else:  # New filename
            if date:
                filename += '_' + time.strftime("%Y%m%d-%H:%M:%S")
            # Check if filename does not already exist. If it does, change name.
            while os.path.exists(filename + '.pt') and len(filename) < 100:
                filename += str(randint(0, 9))
            self._filename = filename
            torch.save(self, self._filename + '.pt')
        return self

    def _load(self, filename, device=torch.device('cpu')):
        try:
            self = torch.load(filename, map_location=device)
        except FileNotFoundError:
            self = torch.load(filename + '.pt', map_location=device)
        return self

    def _to_dict(self):
        "Returns a dict (it's recursive)"
        return_dict = {}
        for k, v in self.__dict__.items():
            if isinstance(v, type(self)):
                return_dict[k] = v._to_dict()
            else:
                return_dict[k] = v
        return return_dict

    def _flatten(self, parent_key='', sep='_'):
        "Recursively flattens nested ddicts"
        d = self._to_dict()
        return shelf._flatten_dict(d)


def log_to_dict(keys_to_log, scope, key_prefix=''):
    """
    Examples::
        >>> a,b = 1.0, 2.0
        >>> d = log_to_dict(['a', 'b'], d, locals())
        >>> d
        >>>     {'a': 1.0, 'b': 2.0}
    """
    d = dict()
    for k in keys_to_log:
        v = scope[k]
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu()  # get out of autograd
            v = np.array(v, dtype=np.float)
        d[key_prefix + k] = v
    return d


def load_descent_data(filename, device, keys):
    CP = shelf()._load(filename, device=device)

    # Load all keys
    ret = []
    for k in keys:
        if k in CP:
            ret.append(CP[k])
        else:
            ret.append(None)

    print('Loaded checkpoint')
    return ret


def avg_iterable(iterable, func):
    '''Applies function `func` to each element of `iterable` and averages the results

        Args:
            iterable: an iterable
            func: function being applied on each element of `iterable`

        Returns:
            Average of `func` applied on `iterable`
    '''
    lst = [func(it) for it in iterable]
    return [sum(x) / len(lst) for x in zip(*lst)]


def clip_norm_(x, max_norm):
    clip_coef = float(max_norm) / (x.norm() + 1e-6)
    if clip_coef > 1:
        x.mul_(clip_coef)
    return x


class Whitener(object):
    def __init__(self, bias, sd, eps=1e-6):
        if type(bias) == torch.Tensor:
            bias = bias.detach().cpu().float().numpy()
        if type(sd) == torch.Tensor:
            sd = sd.detach().cpu().float().numpy()

        self.bias = bias
        self.sd = sd
        self.eps = eps

    def __call__(self, x):
        bias, sd = self.bias, self.sd
        if type(x) == torch.Tensor:
            bias = torch.tensor(bias, device=x.device)
            sd = torch.tensor(sd, device=x.device)
        return (x - bias) / (sd + self.eps)

    def dewhiten(self, y):
        bias, sd = self.bias, self.sd
        if type(y) == torch.Tensor:
            bias = torch.tensor(bias, device=y.device)
            sd = torch.tensor(sd, device=y.device)
        return y * (sd + self.eps) + bias


def plot_weighted_scatter(ax, X, weights=None, color_ind=0, markersize=3.0):
    if weights is None:
        weights = np.ones(len(X)) / len(X)
    rgba_colors = np.zeros((len(X),4))
    rgba_colors[:,color_ind] = 1.0

    rgba_colors[:,3] = weights / np.max(weights) # max(1.0, weights.max())
    ax.scatter(X[:,0], X[:,1], color=rgba_colors, marker='.', s=markersize)
    ax.axis('off')
    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))


def save_animation(seq_points, seq_mmd, seq_weights=None, caption=None, filename=None):
    '''Saves an animation (requires ffmpeg)
    '''
    fig, ax = plt.subplots()
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.axis('off')
    if caption:
        ax.text(0.5,-0.1, caption, size=12, ha="center", transform=ax.transAxes)
    scat = ax.scatter([], [], facecolor='r', marker='.')
    ax.set_title

    def animate(t):
        scat.set_offsets(seq_points[t])
        if seq_weights:
            rgba_colors = np.zeros((len(seq_points[t]), 4))
            rgba_colors[:,0] = 1.0
            rgba_colors[:,3] = seq_weights[t] / max(1.0, seq_weights[t].max())
            scat.set_color(rgba_colors)
        ax.set_title(f'{t}/{len(seq_points)-1}\nmmd = {seq_mmd[t]:.4f}')

    anim = animation.FuncAnimation(fig, animate, frames=len(seq_points), interval=10)
    if filename:
        anim.save(filename + '.mp4')
    return anim


def save_plots(seq_points, seq_mmd, plot_times, targetP, wP=None, seq_weights=None, caption=None, filename=None, alpha_target=0.5, markersize=3.0,
               show_target=True, show_mmd=True):
    """Interpolation plots
    """
    numSub = len(plot_times) + 1
    fig, axs = plt.subplots(ncols=numSub, figsize=(3.5 * numSub, 3))
    for i, (t, ax) in enumerate(zip(plot_times, axs[:-1])):
        if seq_weights is None:
            plot_weighted_scatter(ax, seq_points[t], 1.0, color_ind=0, markersize=markersize)
        else:
            plot_weighted_scatter(ax, seq_points[t], seq_weights[t], color_ind=0, markersize=markersize)

        if show_mmd:
            ax.set_title(f't={t}/{len(seq_points)-1}\nmmd = {seq_mmd[t]:.4f}', y=1.10)
        else:
            ax.set_title(f't={t}/{len(seq_points)-1}', y=1.10)

        ax.axis('off')
        ax.set_xlim((-1, 1))
        ax.set_ylim((-1, 1))
        if i==0 and caption:
            ax.text(0.5,-0.1, caption, size=12, ha="center", transform=ax.transAxes)

    # target
    if show_target:
        ax = axs[-1]
        if wP is None:
            plot_weighted_scatter(ax, targetP.cpu().numpy(), 1.0, color_ind=2, markersize=markersize)
        else:
            plot_weighted_scatter(ax, targetP.cpu().numpy(), wP.view(-1).cpu().numpy(), color_ind=2, markersize=markersize)

        ax.set_title('target', y=1.10)
        ax.axis('off')
        ax.set_xlim((-1, 1))
        ax.set_ylim((-1, 1))

    if filename:
        plt.savefig(filename + '.png', bbox_inches='tight')
    return fig
