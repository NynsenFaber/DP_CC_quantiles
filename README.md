# Differentially Private Quantiles with Smaller Error

To run the code, you need to install a conda environment using the environment.yml file
    
    conda env create -f environment.yml

Then, activate the environment

    conda activate DP_CC_Quantiles

### Benchmarks
1. `DP_AQ` from: *Kaplan, Haim, Shachar Schnapp, and Uri Stemmer. "Differentially private approximate quantiles." International Conference on Machine Learning. PMLR, 2022.*

Used the code from the authors, which is available at: https://github.com/ShacharSchnapp/DP_AQ. The folder `DP_AQ` folder contains a fork of this repository, incorporating minor modifications for compatibility with the experimental setup.

### Experiments
The `real_data.ipynb` notebook contains the experiments on the Adult Datasets.