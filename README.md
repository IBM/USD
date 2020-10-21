# Unbalanced Sobolev Descent

Pytorch source code for paper
> Youssef Mroueh and Mattia Rigotti, "Unbalanced Sobolev Descent", in Advances in Neural Information Processing Systems 33 (NeurIPS), Dec. 2020 [[arXiv:2009.14148](https://arxiv.org/abs/2009.14148)]


## Requirements
* Python 3.6 or above
* PyTorch 1.6.0
* Numpy 1.19.1
* SciPy 1.5.2
* Matplotlib 3.3.1
* PIL 7.2.0

These can be installed using `pip` by running:

```bash
>> pip install -r requirements.txt
```

## Usage

### Synthetic data experiments

To reproduce the the flow simulations between synthetic distributions (Figs 1, 2, 5 and 6) first run the bash script `experiments.bash`, which calls the main python scripts with the appropriate paramters to reproduce the figure:

```bash
>> bash experiments.bash
```
The results will be saved in the folder `final_outputs` and will be used by the notebook `plot_synthetic_data.ipynb` to generate the Figures 1, 2, 5 and 6 in the paper.

 <img src="/figs/syn_cat2heart.png" width="800">  
 <img src="/figs/syn_cat2heart_mmd.png" width="360">


### Interpolation analysis of scRNA-seq data

The notebook `wot_comparison.ipynb` reproduces the interpolation analysis of single-cell RNA sequencing data and generates the relative plots (Figs 4 and 8 in the paper). Please, refer to the instruction in the notebook to download and prepare the data that is used. 

 <img src="/figs/wot.png" width="400">  

## Documents
* [NeurIPS 2020 poster slides](https://github.com/IBM/USD/raw/master/docs/neurips2020_slides.pptx)

## Citation
> Youssef Mroueh, Mattia Rigotti, "Unbalanced Sobolev Descent", in Advances in Neural Information Processing Systems 33 (NeurIPS), Dec. 2020 [[arXiv](https://arxiv.org/abs/2009.14148)]
