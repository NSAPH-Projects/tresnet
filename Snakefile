assert len(config) > 0, "config must be specified with -configfile conf/snakemake.yaml"
assert "experiment" in config, "experiment must be specified with -C experiment=..."

exp = config["experiment"]
params = config["experiments"][exp]
shift_max = params.get("shift_max", 0.25)


# a rule to make to exclude _ from wildcard pattern matching
wildcard_constraints:
    sample="[a-zA-Z0-9-]+",


rule all:
    """
    The purpose of this rule is to assess whether or not the targeted regularization
    is working. For that goal we need to compare against the unregularized case.
    We can use the density ratio classifier and gaussian likelihood only.
    """
    input:
        expand(
            f"logs/{exp}/gaussian/"
            "{seed}/{dset}_{arch}_{strat}_{bb}/srf_estimates.csv",
            seed=list(range(params["num_seeds"])),
            dset=params["datasets"],
            arch=params["architectures"],
            strat=params["strategies"],
            bb=params["backbones"],
        ),


rule impl:
    output:
        f"logs/{exp}/gaussian/" "{seed}/{dset}_{arch}_{strat}_{bb}/srf_estimates.csv",
    log:
        err=f"logs/{exp}/gaussian/" "{seed}/{dset}_{arch}_{strat}_{bb}/stderr.log",
    conda:
        "requirements.yaml"
    # set resources to only use one core per task
    resources:
        cpus=1,
        tasks=1,
        cpus_per_task=1,
        nodes=1,
    shell:
        # f"{python_cmd} main.py"
        "python main.py"
        " dataset={wildcards.dset}"
        " architecture={wildcards.arch}"
        " strategy={wildcards.strat}"
        " seed={wildcards.seed}"
        " outcome.backbone={wildcards.bb}"
        f" logdir=logs/{exp}"
        ' subdir=""'
        " loggers.tb=false"
        " training.progbar=false"
        f" shift.max={shift_max}"
        " &2> {log.err}"
        " && wait"
        # ^ note! without this, it exits early and fails
