# This code is doing shape morphing with Neural Sobolev Descent.
# Modified by mr from the pytorch code accompanying the NIPS 2018 submission
# - added birth/death process
import time
import argparse
import json
import torch
from torch.autograd import grad

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import ddict, get_devices, shelf, save_animation, save_plots
from utils import plot_weighted_scatter, load_descent_data
from datasets import load_data, load_data_weights, minibatch, get_loader
from modules import D_mlp, D_forward_weights, MMD_RFF, manual_sgd_


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unbalanced descent')
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
    parser.add_argument('--save_animation', action='store_true', default=True,
                        help='save animation')
    parser.add_argument('--opts', type=str, default='{}', metavar='OPTS',
                        help='Addition options as dictionary, e.g. {"alpha": 0.1}')
    args = parser.parse_args()


    opt = ddict(
        expName = 'unbalanced',
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
        alpha = 0.6,
        lambda_aug_init = 1e-5,
        rho = 1e-6,
        keep_order = False,
        balance = True, # Whether the number of points in the population is constant
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
        T = args.T,
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


def train_unbalanced_descent(D, dataQ0, dataP, wP, opt):
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
    collQ, coll_mmd = [], []
    birth_total, death_total = 0, 0

    dataQ = dataQ0.clone()
    for t in range(opt.T + 1):
        tic = time.time()

        # Snapshot of current state
        with torch.no_grad():
            mmd_PQ = mmd(dataP, dataQ, weights_X=wP if wP is not None else None)
        coll_mmd.append(mmd_PQ)
        collQ.append(dataQ.detach().cpu().numpy()) # snapshot of current state

        # (1) Update D network
        optimizerD = torch.optim.Adam(D.parameters(), lr=opt.lrD, weight_decay=opt.wdecay, amsgrad=True)
        D.train()
        for i in range(opt.n_c_startup if t==0 else opt.n_c):
            optimizerD.zero_grad()

            x_p, w_p = minibatch((dataP, wP), opt.batchSizeD)
            x_q = minibatch(dataQ, opt.batchSizeD).requires_grad_(True)

            loss, Ep_f, Eq_f, normgrad_f2_q = D_forward_weights(D, x_p, w_p, x_q, 1.0, lambda_aug, opt.alpha, opt.rho)
            loss.backward()
            optimizerD.step()

            manual_sgd_(lambda_aug, opt.rho)

        tocD = time.time() - tic

        # (2) Update Q distribution (with birth/death)
        D.eval()

        # compute initial m_f
        with torch.no_grad():
            x_q = minibatch(dataQ)
            m_f = D(x_q).mean()

        # Update particles positions, and compute birth-death scores
        new_x_q, b_j = [], []
        for x_q, in get_loader(dataQ, batch_size=opt.batchSizeQ):
            x_q = x_q.detach().requires_grad_(True)
            sum_f_q = D(x_q).sum()
            grad_x_q = grad(outputs=sum_f_q, inputs=x_q, create_graph=True)[0]

            with torch.no_grad():
                new_x_q.append(x_q + opt.lrQ * grad_x_q)
                f_q_new = D(new_x_q[-1])

                # birth-death score
                m_f = m_f + (1 / n_samples) * (f_q_new.sum() - sum_f_q)

                b_j.append(f_q_new.view(-1) - m_f)

        new_x_q = torch.cat(new_x_q)
        b_j = torch.cat(b_j)

        # Birth
        idx_alive = (b_j > 0).nonzero().view(-1)
        p_j = 1 - torch.exp(-opt.alpha * opt.tau * b_j[idx_alive])
        idx_birth = idx_alive[p_j > torch.rand_like(p_j)]

        # Death
        idx_neg = (b_j <= 0).nonzero().view(-1)
        p_j = 1 - torch.exp(-opt.alpha * opt.tau * torch.abs(b_j[idx_neg]))
        ix_die = p_j > torch.rand_like(p_j) # Particles that die
        idx_dead = idx_neg[ix_die]
        idx_notdead = idx_neg[~ix_die] # Particles that don't die

        birth_total += len(idx_birth)
        death_total += len(idx_dead)

        if not opt.keep_order:
            new_x_q.data = new_x_q.data[torch.cat((idx_alive, idx_notdead, idx_birth))]

            # Resize population
            if opt.balance:
                n_l = new_x_q.shape[0]

                if n_l < n_samples: # Randomly double particles
                    r_idx = torch.randint(n_l, (n_samples - n_l,))
                    new_x_q = torch.cat((new_x_q, new_x_q[r_idx]))

                if n_l > n_samples: # Randomly kill particles
                    r_idx = torch.randperm(n_l)[:n_samples] # Particles that should be kept
                    new_x_q = new_x_q[r_idx]

        else:
            # Sample dead samples from cloned ones (if there are any), otherwise sample them from alive
            if len(idx_birth) > 0:
                r_idx = idx_birth[torch.randint(len(idx_birth), (len(idx_dead),))]
            else:
                r_idx = idx_alive[torch.randint(len(idx_alive), (len(idx_dead),))]
            new_x_q.data[idx_dead] = new_x_q.data[r_idx]

        dataQ = new_x_q.data

        # (3) print some stuff
        if t % opt.log_every == 0:
            x_p, w_p = minibatch((dataP, wP))
            x_q = minibatch(dataQ)
            loss, Ep_f, Eq_f, normgrad_f2_q = D_forward_weights(D, x_p, w_p, x_q, 1.0, lambda_aug, opt.alpha, opt.rho)
            with torch.no_grad():
                SobDist_lasti = Ep_f.item() - Eq_f.item()
                mmd_dist = mmd(dataP, dataQ, weights_X=wP if wP is not None else None)

            print('[{:5d}/{}] SobolevDist={:.4f}\t mmd={:.5f} births={} deaths={} Eq_normgrad_f2[stepQ]={:.3f} Ep_f={:.2f} Eq_f={:.2f} lambda_aug={:.4f}'.\
                format(t, opt.T, SobDist_lasti, mmd_dist, birth_total, death_total, normgrad_f2_q.mean().item(), Ep_f.item(), Eq_f.item(), lambda_aug.item()))

            if opt.plot_online:
                line.set_data(dataQ[:,0].detach().cpu().numpy(), dataQ[:,1].detach().cpu().numpy())
                plt.pause(0.01)

    return dataQ, collQ, coll_mmd


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
    dataQ, collQ, coll_mmd = train_unbalanced_descent(D, dataQ0, dataP, wP, opt)

    # Save
    CP = shelf(dataQ=dataQ, dataP=dataP, wP=wP, collQ=collQ, coll_mmd=coll_mmd, opt=opt)
    CP._save(FILENAME, date=False)

    # Save plot
    save_plots(collQ, coll_mmd, opt.plottimes, dataP, wP, filename=FILENAME, alpha_target=0.2,
            caption=r"$\tau=${}, $\alpha=${}, bsD={}, $n_c$={}, layers={}".format(opt.tau, opt.alpha, opt.batchSizeD, opt.n_c, opt.layerSizes))

    # Save animation
    if opt.save_animation:
        save_animation(collQ, coll_mmd, caption=r"$\tau=${}, $\alpha=${}".format(opt.tau, opt.alpha), filename=FILENAME)
