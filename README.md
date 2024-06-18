# **T**argeted **R**egularization for **E**xposure **S**hifts with Neural **Net**works (TRESNet)

This is the official PyTorch implementation of our paper:

```bibtex
@inproceedings{tec2023causal,
  title={Causal Estimation of Exposure Shifts with Neural Networks},
  author={Tec, Mauricio and Mudele, Oladimeji and Josey, Kevin and Dominici, Francesca},
  booktitle={KDD},
  year={2024}
}
```

# Reproducing the paper results 

## Benchmarks

For the benchmarks, simply run `bash scripts/benchmarks.sh` from the root folder. 
To visualize the results, use the notebook `notebooks/benchmark_results.ipynb`.

Currently, the script defaults to 24 parallel CPU processes and no GPUs. Those settings can be easily changed by changing the first two lines in the script to set the variables, `num_parallel` and `num_gpus`.

## Application

Our Medicare dataset is provided by [ResDAC](https://resdac.org/). It is not publicly available due to licensing. Notice that working with Medicare data requires a Research, Ethics and Compliance Training (CITI), an Institutional Review Board (IRB), and a data usage agreement (DUA), which the authors have. 

For replicability purposes, we have provided the script `scripts/medicare.sh` used to produce the paper results. This script calls `main_medicare.py`, which in turn uses the same codebase and structure than `main.py` used for the paper benchmarks. The paper plots were produced with the notebook `notebooks/medicare_results.ipynb`.


# Files 

* `main.py`. Main Python file to run the tresnet experiments, except those using ipw.
* `main_medicare.py`. Main Python file to run the application results.
* `tresnet/`
    - `models.py`. Defines the base architectures (VCNet and DRnet) used for tresnet.
    - `modules.py`. Auxiliary neural network layers.
    - `ratios.py`. Utils to compute log density ratios and other regularization methods not used in the paper.
* `dataset/`
    - `ihdp/ihdp.csv`
    - `news/news_preprocessed.npy`. The processed file is copied from the [VCNet repo](https://github.com/lushleaf/varying-coefficient-net-with-functional-tr).
    - `datasets.py`. Defines the synthetic benchmark datasets loaded on demand.
    - `medicare.py`. Loads the application dataset and prepares it for deep learning.
* `notebooks/`
    - `benchmark_results.ipynb`. Constructs the paper benchmark table.
    - `medicare_results.ipynb`. Generates the application plots from the results.
    - `example_figures.py`. Generates the figures for the toy air pollution example in the paper.
* `scripts/`
    - `benchmarks.sh`. Runs all benchmarks (see above).
    - `medicare.sh`. Trains the model with the application dataset.
