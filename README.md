# TRESNET

This is the official implementation of our paper

```bibtex
@inproceedings{tec2023causal,
  title={Causal Estimation of Exposure Shifts with Neural Networks},
  author={Tec, Mauricio and Mudele, Oladimeji and Josey, Kevin and Dominici, Francesca},
  booktitle={KDD},
  year={2024}
}
```

### Experiments

We use the [Snakemake](https://snakemake.readthedocs.io/en/stable/index.html) pipeline system for reproducibility. 

The install the dependencies, run
```bash
conda env create -f requirements.yaml  # defines the tresnet conda env
```
The code to reproduce all experiments is in the `Snakefile`. At a high level, the code runs using the following command
```bash
source activate tresnet
snakemake --cores <num-cores> -C experiment=<exp> strategy=<strategy>
```
where `<exp>` is the experiment number (see the paper for experiment's definition), `<strategy>` is the causal estimation method. The config file `conf/snakemake.yaml` defines all the possible strategies per experiment.

Snakemake is running the main training script in `main.py` with the following pattern
```bash
python main.py
    dataset=<indp/news/...>
    strategy=<tresnet/aipw/...>
    outcome.backbone=<piecewise/vc>
    family=<gaussian/poisson/bernoulli>
    seed=<seed>
```
We use the [Hydra](https://hydra.cc/) configuration system to manage the hyperparameters. The `conf/` folder contains the configuration files for each experiment. The `conf/config.yaml` file contains the default hyperparameters for the models. 

The `tresnet/` contains all the auxiliary code such as the network architectures and utilies. Additionally, the benchmark datasets are constructed in the module `tresnet/datamodules/`.

See the folder `slurm/` folder for example scripts to run the experiments using snakemake. We used these scripts for the paper experiments in a cluster. 

