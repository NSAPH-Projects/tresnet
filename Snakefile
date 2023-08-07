# python command when in sbatch
conda: "requirements.yaml"


python_cmd = "srun -n 1 -c 1 python" if config["use_srun"] else "python"


# add a directive to exclude _ from wildcard matching
wildcard_constraints:
    sample="[a-zA-Z0-9-]+",


rule exp1_all:
    """
    The purpose of this rule is to assess whether or not the targeted regularization
    is working. For that goal we need to compare against the unregularized case.
    We can use the density ratio classifier and gaussian likelihood only.
    """
    input:
        expand(
            "logs/exp1/gaussian/{seed}/{dset}_{arch}_{strat}_{bb}/srf_estimates.csv",
            seed=list(range(config["exp1"]["num_seeds"])),
            dset=config["exp1"]["datasets"],
            arch=config["exp1"]["architectures"],
            strat=config["exp1"]["strategies"],
            bb=config["exp1"]["backbones"],
        ),


rule exp1_impl:
    output:
        "logs/exp1/gaussian/{seed}/{dset}_{arch}_{strat}_{bb}/srf_estimates.csv",
    log:
        err="logs/exp1/gaussian/{seed}/{dset}_{arch}_{strat}_{bb}/stderr.log",
    shell:
        f"{python_cmd} main.py"
        " dataset={wildcards.dset}"
        " architecture={wildcards.arch}"
        " strategy={wildcards.strat}"
        " seed={wildcards.seed}"
        " outcome.backbone={wildcards.bb}"
        " logdir=logs/exp1"
        ' subdir=""'
        " loggers.tb=false"
        " training.progbar=false"
        " &2> {log.err}"
        " && wait"
        # ^ note! without this, it exits early and fails
