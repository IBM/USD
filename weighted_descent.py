# This code is doing shape morphing with Neural Sobolev Descent.
# Modified by mr from the pytorch code accompanying the NIPS 2018 submission
# - added birth/death process
import time
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import ddict, get_devices, shelf, save_animation, save_plots
from utils import plot_weighted_scatter, load_descent_data, clip_norm_
from datasets import load_data, load_data_weights, minibatch, get_loader
from modules import D_mlp, D_forward_weights, MMD_RFF, manual_sgd_


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Weighted descent')
    parser.add_argument('--n_hiddens', type=int, default=1024, metavar='N',
                        help='number of hidden neurons')
    parser.add_argument('--T', type=int, default=800, metavar='T',
                        help='time steps')
    parser.add_argument('--src', type=str, default='gauss', metavar='S',
                        help='source')
    parser.add_argument('--target', type=str, default='mog', metavar='T',
                        help='target')
    parser.add_argument('--target_weights', type=str, default=None, metavar='W',
                        help='target')
    parser.add_argument('--save_animation', action='store_true', default=False,
                        help='save animation')
    parser.add_argument('--opts', type=str, default='{}', metavar='OPTS',
                        help='Addition options as dictionary, e.g. {"alpha": 0.1}')
    args = parser.parse_args()


    # Default parameters
    opt = ddict(
        expName = 'weighted',
        expDirName = 'output',
        batchSizeD = 512,
        batchSizeQ = 1000,
        n_c = 20,
        n_c_startup = 200,
        nonlin = 'ReLU',
        normalization = None,
        dropout = 0.2,
        wdecay = 1e-5,
        lrD = 1e-4,
        lrQ = 1e-4,
        tau = 1e-3, # Birth/Death (should be the order as lrQ)
        alpha = 0.5,
        lambda_aug_init = 1e-5,
        rho = 1e-6,
        T = args.T,
        n_plots = 5,
        log_every = 10,
        nPointsMax_src = 4000,
        nPointsMax_target = 4000,
        seed = 123,
        plot_online = False,
        no_cuda = False
    )

    # Replace default paramenters with those given at terminal
    opt.update(ddict(
        src = args.src,
        target = args.target,
        target_weights = args.target_weights,
        layerSizes = [64, args.n_hiddens, 64],
        save_animation = args.save_animation))

    # Parse --opts
    additional_opts = json.loads(args.opts.replace('\n', ''))
    opt.update(additional_opts)

    # Plot times exponentially distributed
    opt.plottimes = [opt.T // 2**n for n in range(opt.n_plots - 1)]
    opt.plottimes = [0] + opt.plottimes[::-1]


def train_weighted_descent(D, dataQ0, dataP, wP, opt):
    n_samples, n_features = dataQ0.shape
    device = dataQ0.device

    # Lagrange multiplier for Augmented Lagrangian
    lambda_aug = torch.tensor([opt.lambda_aug_init], requires_grad=True, device=device)

    # MMD distance
    mmd = MMD_RFF(num_features=n_features, num_outputs=300).to(device)

    # Train
    print('Start training')

    if opt.plot_online:
        fig, ax = plt.subplots()
        ax.set_xlim((-1.1, 1.1))
        ax.set_ylim((-1.1, 1.1))
        scat = ax.scatter([], [], facecolor='r')

    # Save stuff
    wQ = torch.ones((len(dataQ0), 1), device=device)
    collQ, collW, coll_mmd = [], [], []

    dataQ = dataQ0.clone()
    for t in range(opt.T + 1):
        tic = time.time()

        # Snapshot of current state
        with torch.no_grad():
            mmd_PQ = mmd(dataP, dataQ, weights_X=wP if wP is not None else None, weights_Y=wQ)

        coll_mmd.append(mmd_PQ)
        collQ.append(dataQ.detach().cpu().numpy()) # snapshot of current state
        collW.append(wQ.view(-1).detach().cpu().numpy()) # snapshot of current weights

        # (1) Update D network
        optimizerD = torch.optim.Adam(D.parameters(), lr=opt.lrD, weight_decay=opt.wdecay, amsgrad=True)
        D.train()
        for i in range(opt.n_c_startup if t==0 else opt.n_c):
            optimizerD.zero_grad()

            x_p, w_p = minibatch((dataP, wP), opt.batchSizeD)
            x_q, w_q = minibatch((dataQ, wQ), opt.batchSizeD)

            loss, Ep_f, Eq_f, normgrad_f2_q = D_forward_weights(D, x_p, w_p, x_q, w_q, lambda_aug, opt.alpha, opt.rho)
            loss.backward()
            optimizerD.step()

            manual_sgd_(lambda_aug, opt.rho)

        tocD = time.time() - tic

        # (2) Update Q distribution (with birth/death)
        D.eval()
        with torch.no_grad():
            x_q, w_q = minibatch((dataQ, wQ))
            f_q = D(x_q)
            m_f = (w_q * f_q).mean()

        new_x_q, log_wQ = [], []
        for x_q, w_q in get_loader((dataQ, wQ), batch_size=opt.batchSizeQ):
            x_q = x_q.detach().requires_grad_(True)
            sum_f_q = D(x_q).sum()
            grad_x_q = grad(outputs=sum_f_q, inputs=x_q, create_graph=True)[0]

            # Update particles
            with torch.no_grad():
                # Move particles
                x_q.data += opt.lrQ * grad_x_q
                f_q = D(x_q)
                dw_q = f_q - m_f

                log_wQ.append((w_q / n_samples).log() + opt.tau * dw_q)
                new_x_q.append(x_q)

        # Update weights and dataQ
        wQ = F.softmax(torch.cat(log_wQ), dim=0) * n_samples
        dataQ = torch.cat(new_x_q)

        # (3) print some stuff
        if t % opt.log_every == 0:
            x_p, w_p = minibatch((dataP, wP))
            x_q, w_q = minibatch((dataQ, wQ))

            loss, Ep_f, Eq_f, normgrad_f2_q = D_forward_weights(D, x_p, w_p, x_q, w_q, lambda_aug, opt.alpha, opt.rho)
            with torch.no_grad():
                SobDist_lasti = Ep_f.item() - Eq_f.item()
                mmd_dist = mmd(dataP, dataQ, weights_X=wP if wP is not None else None, weights_Y=wQ)

            print('[{:5d}/{}] SobolevDist={:.4f}\t mmd={:.5f} Eq_normgrad_f2[stepQ]={:.3f} Ep_f={:.2f} Eq_f={:.2f} lambda_aug={:.4f}'.\
                format(t, opt.T, SobDist_lasti, mmd_dist, normgrad_f2_q.mean().item(), Ep_f.item(), Eq_f.item(), lambda_aug.item()))

            if opt.plot_online:
                scat.set_offsets(dataQ.detach().cpu().numpy())
                rgba_colors = np.zeros((wQ.shape[0],4))
                rgba_colors[:,0] = 1.0
                rgba_colors[:,3] = wQ.view(-1).detach().cpu().numpy() / wQ.max().item()
                scat.set_color(rgba_colors)
                plt.pause(0.01)

    return dataQ, wQ, collQ, collW, coll_mmd


if __name__ == "__main__":
    FILENAME = opt.expDirName + '/' + opt.expName + '_'
    FILENAME += opt.src.split('/')[-1][:3] + '-' + opt.target.split('/')[-1][:3]

    device, num_gpus = get_devices("cuda:0" if not opt.no_cuda and torch.cuda.is_available() else "cpu", seed=opt.seed)

    # Load data
    dataQ0 = load_data(opt.src, opt.nPointsMax_src).to(device)
    dataP, wP = load_data_weights(opt.target, opt.target_weights, opt.nPointsMax_target)
    dataP = dataP.to(device)
    if wP is not None:
        wP = wP.to(device)

    n_samples, n_features = dataQ0.shape

    print(opt)

    # Discriminator
    D = D_mlp(n_features, opt.layerSizes, opt.nonlin, opt.normalization, opt.dropout).to(device)
    print(D)

    # Run descent
    dataQ, wQ, collQ, collW, coll_mmd = train_weighted_descent(D, dataQ0, dataP, wP, opt)

    # Save
    CP = shelf(dataQ=dataQ, dataP=dataP, wQ=wQ, wP=wP, collQ=collQ, collW=collW, coll_mmd=coll_mmd, opt=opt)
    CP._save(FILENAME, date=False)

    # Save plot
    save_plots(collQ, coll_mmd, opt.plottimes, dataP, wP, collW, filename=FILENAME, alpha_target=0.2,
            caption=r"$\tau=${}, $\alpha=${}, bsD={}, $n_c$={}, layers={}".format(opt.tau, opt.alpha, opt.batchSizeD, opt.n_c, opt.layerSizes))

    # Save animation
    if opt.save_animation:
        save_animation(collQ, coll_mmd, collW, r"$\tau=${}, $\alpha=${}".format(opt.tau, opt.alpha), FILENAME)
