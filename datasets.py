import torch
from torch import nn
import numpy as np
from PIL import Image
from scipy.spatial import distance


def load_img(fn='img/heart.jpg', size=200, max_samples=None):
    r"""Returns x,y of black pixels (between -1 and 1)
    """
    pic = np.array(Image.open(fn).resize((size,size)).convert('L'))
    y_inv, x = np.nonzero(pic<=128)
    y = size - y_inv - 1
    if max_samples and x.size > max_samples:
        ixsel = np.random.choice(x.size, max_samples, replace=False)
        x, y = x[ixsel], y[ixsel]
    return np.stack((x, y), 1) / size * 2 - 1


def load_weights(pos, img='img/spiral3d.jpg', size=200):
    if type(pos) is torch.Tensor:
        pos = pos.detach().cpu().numpy()
    pos_int = ((pos + 1)  / 2 * size).astype(int)

    # Reverse y
    pos_int[:,1] = size - pos_int[:,1] - 1

    weights_pic = np.array(Image.open(img).resize((size,size)).convert('L'))
    weights = 255 - weights_pic[pos_int[:,1], pos_int[:,0]]
    weights = (weights - weights.min()) / (weights.max() - weights.min())
    return weights


def describe_data(D):
    """Prints size, min, max, mean and std of a matrix (numpy.ndarray or torch.Tensor)
    """
    s = '{:8s} [{:.4f} , {:.4f}], m+-s = {:.4f} +- {:.4f}'
    si = 'x'.join(map(str, D.shape))
    if isinstance(D, torch.Tensor):
        vals = D.min().item(), D.max().item(), D.mean().item(), D.std().item()
    else:
        vals = D.min(), D.max(), D.mean(), D.std()
    return s.format(si, *vals)


def generate_mog_data(num_modes=8, radius=0.75, center=(0, 0), sigma=0.075, size_class=1000):
    r"""Generated Mixture of Gaussian dataset

    Example:
    >>> import matplotlib.pyplot as plt
    >>> data = generate_mog_data()
    >>> plt.scatter(data[:,0], data[:,1], alpha=0.1)

    """
    total_data = {}

    t = np.linspace(0, 2*np.pi, num_modes+1)
    t = t[:-1]
    x = np.cos(t)*radius + center[0]
    y = np.sin(t)*radius + center[1]

    modes = np.vstack([x, y]).T

    for idx, mode in enumerate(modes):
        x = np.random.normal(mode[0], sigma, size_class)
        y = np.random.normal(mode[1], sigma, size_class)
        total_data[idx] = np.vstack([x, y]).T

    all_points = np.vstack([values for values in total_data.values()])
    all_points = np.random.permutation(all_points)[0:size_class * num_modes]
    return all_points


def generate_circles_data(n_samples=1000, noise=None, factor=.8):
    """Make a large circle containing a smaller circle in 2d.

    Parameters
    ----------
    n_samples : int, optional (default=1000)
        Total number of points for both circles
    noise : double or None (default=None)
        Standard deviation of Gaussian noise added to the data.
    factor : 0 < double < 1 (default=.8)
        Scale factor between inner and outer circle.
    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.
    y : array of shape [n_samples]
        The integer labels (0 or 1) for class membership of each sample.
    """

    if factor >= 1 or factor < 0:
        raise ValueError("'factor' has to be between 0 and 1.")

    n_samples_outer = (n_samples + 1) // 2
    n_samples_inner = n_samples // 2

    linspace_outer = np.linspace(0, 2 * np.pi, n_samples_outer, endpoint=False)
    linspace_inner = np.linspace(0, 2 * np.pi, n_samples_inner, endpoint=False)
    outer_circ_x = np.cos(linspace_outer)
    outer_circ_y = np.sin(linspace_outer)
    inner_circ_x = np.cos(linspace_inner) * factor
    inner_circ_y = np.sin(linspace_inner) * factor

    X = np.vstack([np.append(outer_circ_x, inner_circ_x),
                   np.append(outer_circ_y, inner_circ_y)]).T
    y = np.hstack([np.zeros(n_samples_outer, dtype=np.intp),
                   np.ones(n_samples_inner, dtype=np.intp)])

    if noise is not None:
        X += np.random.normal(0.0, noise, size=X.shape)

    return X, y


def minibatch(data, batch_size=None):
    if type(data) is not tuple:
        data =(data,)

    if batch_size:
        idx = torch.randperm(data[0].shape[0])[:batch_size]
        out = [d[idx] if d is not None else None for d in data]
    else:
        out = [d.clone() if d is not None else None for d in data]

    if len(out) == 1:
        return out[0]
    else:
        return out


def get_loader(data, batch_size, shuffle=False):
    if type(data) is not tuple:
        data =(data,)
    ds = torch.utils.data.TensorDataset(*data)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def load_data(filename, n_points_max):
    # Load data
    if filename.split('.')[-1] in ['png', 'jpg']:
        data = load_img(filename, max_samples=n_points_max)
    elif filename== 'mog':
        data = generate_mog_data(size_class=n_points_max//8)
    elif filename == 'gauss':
        data = generate_mog_data(num_modes=1, radius=0, sigma=0.1, size_class=n_points_max)
    elif filename == 'circles':
        data = 0.9 * generate_circles_data(n_points_max, noise=0.05, factor=0.5)[0]
    else:
        raise ValueError('File not found')

    data = torch.from_numpy(data).float()
    print('data', describe_data(data))
    return data


def load_data_weights(filename, filename_weights, n_points_max):
    dataP = load_data(filename, n_points_max)
    if filename_weights is None:
        weightsP = None
    else:
        weightsP = torch.from_numpy(load_weights(dataP, img=filename_weights)).float()
        weightsP = weightsP.view(-1, 1)
        weightsP = weightsP * len(weightsP) / weightsP.sum()
    return dataP, weightsP


def train(model, criterion, train_loader, optimizer, log_times=10):
    '''Trains model
    '''
    model.train()
    device = next(model.parameters()).device

    mean_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        mean_loss += loss.item() / len(data)
        if log_times > 0 and batch_idx % (len(train_loader) // log_times) == 0:
            print('   training progress: {}/{} ({:.0f}%)\tloss: {:.6f}'.format(
                batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))

    return mean_loss


def test(model, criterion, data_loader, msg=''):
    '''Compute model accuracy
    '''
    model.eval()
    device = next(model.parameters()).device

    test_loss, correct = 0.0, 0.0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if isinstance(output, tuple):
                output = output[0]
            test_loss += criterion(output, target).item()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()

    accuracy = float(correct) / len(data_loader.dataset)
    test_loss /= len(data_loader)  # loss function already averages over batch size
    if msg:
        print('{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            msg, test_loss, correct, len(data_loader.dataset), 100. * accuracy))
    return accuracy


def test_criterion(model, criterion, data_loader, msg=''):
    '''Compute model accuracy
    '''
    model.eval()
    device = next(model.parameters()).device

    test_loss = 0.0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if isinstance(output, tuple):
                output = output[0]
            test_loss += criterion(output, target).item()

    test_loss /= len(data_loader)  # loss function already averages over batch size
    return test_loss
