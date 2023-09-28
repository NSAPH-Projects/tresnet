# TRESNET

We use the [Snakemake](https://snakemake.readthedocs.io/en/stable/index.html) pipeline system for reproducibility. The necessary prerequisites to run the code can be installed using conda
```bash
conda env create -f requirements.yaml
```

See the folder `slurm/` folder for example scripts to run the experiments using snakemake. We used these scripts for the paper experiments in a cluster.

The code pertaining to the TRESNET architecture and regularization is in the `tresnet/` folder. The benchmark datasets are defined in the module `tresnet/datamodules/`.

