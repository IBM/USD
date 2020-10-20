#!/bin/bash
set -e
set -o pipefail

if [ ! -d final_outputs ]; then
    mkdir final_outputs
fi

# Experiment 1a: gauss -> mog (Sobolev, i.e. tau = 0.0)

if [ ! -f final_outputs/1a_weighted_noW_gau-mog.pt ]; then
    python weighted_descent.py --opts '{
        "expName": "1a_weighted_noW",
        "expDirName": "final_outputs",
        "src": "gauss",
        "target": "mog",
        "target_weights": null,
        "layerSizes": [64, 1024, 64],
        "T": 800,
        "batchSizeD": 512,
        "batchSizeQ": 1000,
        "n_c": 20,
        "n_c_startup": 200,
        "nonlin": "ReLU",
        "normalization": null,
        "dropout": 0.2,
        "wdecay": 1e-5,
        "lrD": 1e-4,
        "lrQ": 1e-4,
        "tau": 0.0,
        "alpha": 0.6,
        "lambda_aug_init": 1e-5,
        "rho": 1e-6,
        "n_plots": 5,
        "log_every": 10,
        "nPointsMax_src": 4000,
        "nPointsMax_target": 4000,
        "seed": 123,
        "save_animation": true,
        "plot_online": false}'
fi

# Experiment 1b: gauss -> mog (Unbalanced Sobolev, i.e. birth-death)
if [ ! -f final_outputs/1b_unbalanced_gau-mog.pt ]; then
    python unbalanced_descent.py --opts '{
        "expName": "1b_unbalanced",
        "expDirName": "final_outputs",
        "src": "gauss",
        "target": "mog",
        "target_weights": null,
        "layerSizes": [64, 1024, 64],
        "T": 800,
        "batchSizeD": 512,
        "batchSizeQ": 1000,
        "n_c": 20,
        "n_c_startup": 200,
        "nonlin": "ReLU",
        "normalization": null,
        "dropout": 0.2,
        "wdecay": 1e-5,
        "lrD": 1e-4,
        "lrQ": 1e-4,
        "tau": 1e-3,
        "balance": true,
        "alpha": 0.6,
        "lambda_aug_init": 1e-5,
        "rho": 1e-6,
        "n_plots": 5,
        "log_every": 10,
        "nPointsMax_src": 4000,
        "nPointsMax_target": 4000,
        "seed": 123,
        "save_animation": true,
        "plot_online": false}'
fi

# Experiment 1c: gauss -> mog (weighted)
if [ ! -f final_outputs/1c_weighted_gau-mog.pt ]; then
    python weighted_descent.py --opts '{
        "expName": "1c_weighted",
        "expDirName": "final_outputs",
        "src": "gauss",
        "target": "mog",
        "target_weights": null,
        "layerSizes": [64, 1024, 64],
        "T": 800,
        "batchSizeD": 512,
        "batchSizeQ": 1000,
        "n_c": 20,
        "n_c_startup": 200,
        "nonlin": "ReLU",
        "normalization": null,
        "dropout": 0.2,
        "wdecay": 1e-5,
        "lrD": 1e-4,
        "lrQ": 1e-4,
        "tau": 1e-3,
        "alpha": 0.6,
        "lambda_aug_init": 1e-5,
        "rho": 1e-6,
        "n_plots": 5,
        "log_every": 10,
        "nPointsMax_src": 4000,
        "nPointsMax_target": 4000,
        "seed": 123,
        "save_animation": true,
        "plot_online": false}'
fi


# Experiment 2a: gauss -> concentric circles (Sobolev, i.e. tau = 0.0)
if [ ! -f final_outputs/2a_weighted_noW_gau-cir.pt ]; then
    python weighted_descent.py --opts '{
        "expName": "2a_weighted_noW",
        "expDirName": "final_outputs",
        "src": "gauss",
        "target": "circles",
        "target_weights": null,
        "layerSizes": [64, 1024, 64],
        "T": 800,
        "batchSizeD": 512,
        "batchSizeQ": 1000,
        "n_c": 20,
        "n_c_startup": 200,
        "nonlin": "ReLU",
        "normalization": null,
        "dropout": 0.2,
        "wdecay": 1e-5,
        "lrD": 1e-4,
        "lrQ": 1e-4,
        "tau": 0.0,
        "alpha": 0.6,
        "lambda_aug_init": 1e-5,
        "rho": 1e-6,
        "n_plots": 5,
        "log_every": 10,
        "nPointsMax_src": 2000,
        "nPointsMax_target": 2000,
        "seed": 123,
        "save_animation": true,
        "plot_online": false}'
fi

# Experiment 2b: gauss -> concentric circles (Unbalanced Sobolev, i.e. birth-death)
if [ ! -f final_outputs/2b_unbalanced_cir-mog.pt ]; then
    python unbalanced_descent.py --opts '{
        "expName": "2b_unbalanced",
        "expDirName": "final_outputs",
        "src": "gauss",
        "target": "circles",
        "target_weights": null,
        "layerSizes": [64, 1024, 64],
        "T": 800,
        "batchSizeD": 512,
        "batchSizeQ": 1000,
        "n_c": 20,
        "n_c_startup": 200,
        "nonlin": "ReLU",
        "normalization": null,
        "dropout": 0.2,
        "wdecay": 1e-5,
        "lrD": 1e-4,
        "lrQ": 1e-4,
        "tau": 1e-3,
        "balance": true,
        "alpha": 0.6,
        "lambda_aug_init": 1e-5,
        "rho": 1e-6,
        "n_plots": 5,
        "log_every": 10,
        "nPointsMax_src": 2000,
        "nPointsMax_target": 2000,
        "seed": 123,
        "save_animation": true,
        "plot_online": false}'
fi

# Experiment 2c: gauss -> concentric circles (weighted)
if [ ! -f final_outputs/2c_weighted_gau-cir.pt ]; then
    python weighted_descent.py --opts '{
        "expName": "2c_weighted",
        "expDirName": "final_outputs",
        "src": "gauss",
        "target": "circles",
        "target_weights": null,
        "layerSizes": [64, 1024, 64],
        "T": 800,
        "batchSizeD": 512,
        "batchSizeQ": 1000,
        "n_c": 20,
        "n_c_startup": 200,
        "nonlin": "ReLU",
        "normalization": null,
        "dropout": 0.2,
        "wdecay": 1e-5,
        "lrD": 1e-4,
        "lrQ": 1e-4,
        "tau": 1e-3,
        "alpha": 0.6,
        "lambda_aug_init": 1e-5,
        "rho": 1e-6,
        "n_plots": 5,
        "log_every": 10,
        "nPointsMax_src": 2000,
        "nPointsMax_target": 2000,
        "seed": 123,
        "save_animation": true,
        "plot_online": false}'
fi


# Experiment 3a: disk -> heart+spiral (Sobolev, i.e. tau = 0.0)
if [ ! -f final_outputs/3a_weighted_noW_dis-hea.pt ]; then
    python weighted_descent.py --opts '{
        "expName": "3a_weighted_noW",
        "expDirName": "final_outputs",
        "src": "img/disk.png",
        "target": "img/heart.png",
        "target_weights": "img/spiral3d.jpg",
        "layerSizes": [64, 1024, 64],
        "T": 800,
        "batchSizeD": 512,
        "batchSizeQ": 1000,
        "n_c": 20,
        "n_c_startup": 200,
        "nonlin": "ReLU",
        "normalization": null,
        "dropout": 0.2,
        "wdecay": 1e-5,
        "lrD": 1e-4,
        "lrQ": 1e-4,
        "tau": 0.0,
        "alpha": 0.8,
        "lambda_aug_init": 1e-5,
        "rho": 1e-6,
        "n_plots": 5,
        "log_every": 10,
        "nPointsMax_src": 4000,
        "nPointsMax_target": 4000,
        "seed": 123,
        "save_animation": true,
        "plot_online": false}'
fi

# Experiment 3b: disk -> heart+spiral (Unbalanced Sobolev, i.e. birth-death)
if [ ! -f final_outputs/3b_unbalanced_dis-hea.pt ]; then
    python unbalanced_descent.py --opts '{
        "expName": "3b_unbalanced",
        "expDirName": "final_outputs",
        "src": "img/disk.png",
        "target": "img/heart.png",
        "target_weights": "img/spiral3d.jpg",
        "layerSizes": [64, 1024, 64],
        "T": 800,
        "batchSizeD": 512,
        "batchSizeQ": 1000,
        "n_c": 20,
        "n_c_startup": 200,
        "nonlin": "ReLU",
        "normalization": null,
        "dropout": 0.2,
        "wdecay": 1e-5,
        "lrD": 1e-4,
        "lrQ": 1e-4,
        "tau": 1e-3,
        "balance": true,
        "alpha": 0.8,
        "lambda_aug_init": 1e-5,
        "rho": 1e-6,
        "n_plots": 5,
        "log_every": 10,
        "nPointsMax_src": 4000,
        "nPointsMax_target": 4000,
        "seed": 123,
        "save_animation": true,
        "plot_online": false}'
fi

# Experiment 3c: disk -> heart+spiral  (weighted)
if [ ! -f final_outputs/3c_weighted_dis-hea.pt ]; then
    python weighted_descent.py --opts '{
        "expName": "3c_weighted",
        "expDirName": "final_outputs",
        "src": "img/disk.png",
        "target": "img/heart.png",
        "target_weights": "img/spiral3d.jpg",
        "layerSizes": [64, 1024, 64],
        "T": 800,
        "batchSizeD": 512,
        "batchSizeQ": 1000,
        "n_c": 20,
        "n_c_startup": 200,
        "nonlin": "ReLU",
        "normalization": null,
        "dropout": 0.2,
        "wdecay": 1e-5,
        "lrD": 1e-4,
        "lrQ": 1e-4,
        "tau": 1e-3,
        "alpha": 0.8,
        "lambda_aug_init": 1e-5,
        "rho": 1e-6,
        "n_plots": 5,
        "log_every": 10,
        "nPointsMax_src": 4000,
        "nPointsMax_target": 4000,
        "seed": 123,
        "save_animation": true,
        "plot_online": false}'
fi


# Experiment 4a: cat -> heart+gradient (Sobolev, i.e. tau = 0.0)
if [ ! -f final_outputs/4a_weighted_noW_cat-hea.pt ]; then
    python weighted_descent.py --opts '{
        "expName": "4a_weighted_noW",
        "expDirName": "final_outputs",
        "src": "img/cat.png",
        "target": "img/heart.png",
        "target_weights": "img/hgradient.png",
        "layerSizes": [64, 1024, 64],
        "T": 800,
        "batchSizeD": 512,
        "batchSizeQ": 1000,
        "n_c": 20,
        "n_c_startup": 200,
        "nonlin": "ReLU",
        "normalization": null,
        "dropout": 0.2,
        "wdecay": 1e-5,
        "lrD": 1e-4,
        "lrQ": 1e-4,
        "tau": 0.0,
        "alpha": 0.8,
        "lambda_aug_init": 1e-5,
        "rho": 1e-6,
        "n_plots": 5,
        "log_every": 10,
        "nPointsMax_src": 4000,
        "nPointsMax_target": 4000,
        "seed": 123,
        "save_animation": true,
        "plot_online": false}'
fi

# Experiment 4b: cat -> heart+gradient (Unbalanced Sobolev, i.e. birth-death)
if [ ! -f final_outputs/4b_unbalanced_cat-hea.pt ]; then
    python unbalanced_descent.py --opts '{
        "expName": "4b_unbalanced",
        "expDirName": "final_outputs",
        "src": "img/cat.png",
        "target": "img/heart.png",
        "target_weights": "img/hgradient.png",
        "layerSizes": [64, 1024, 64],
        "T": 800,
        "batchSizeD": 512,
        "batchSizeQ": 1000,
        "n_c": 20,
        "n_c_startup": 200,
        "nonlin": "ReLU",
        "normalization": null,
        "dropout": 0.2,
        "wdecay": 1e-5,
        "lrD": 1e-4,
        "lrQ": 1e-4,
        "tau": 1e-3,
        "balance": true,
        "alpha": 0.8,
        "lambda_aug_init": 1e-5,
        "rho": 1e-6,
        "n_plots": 5,
        "log_every": 10,
        "nPointsMax_src": 4000,
        "nPointsMax_target": 4000,
        "seed": 123,
        "save_animation": true,
        "plot_online": false}'
fi

# Experiment 4c: cat -> heart+gradient  (weighted)
if [ ! -f final_outputs/4c_weighted_cat-hea.pt ]; then
    python weighted_descent.py --opts '{
        "expName": "4c_weighted",
        "expDirName": "final_outputs",
        "src": "img/cat.png",
        "target": "img/heart.png",
        "target_weights": "img/hgradient.png",
        "layerSizes": [64, 1024, 64],
        "T": 800,
        "batchSizeD": 512,
        "batchSizeQ": 1000,
        "n_c": 20,
        "n_c_startup": 200,
        "nonlin": "ReLU",
        "normalization": null,
        "dropout": 0.2,
        "wdecay": 1e-5,
        "lrD": 1e-4,
        "lrQ": 1e-4,
        "tau": 1e-3,
        "alpha": 0.8,
        "lambda_aug_init": 1e-5,
        "rho": 1e-6,
        "n_plots": 5,
        "log_every": 10,
        "nPointsMax_src": 4000,
        "nPointsMax_target": 4000,
        "seed": 123,
        "save_animation": true,
        "plot_online": false}'
fi
