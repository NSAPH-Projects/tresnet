assert len(config) > 0, "config must be specified with -configfile conf/snakemake.yaml"
assert "experiment" in config, "experiment must be specified with -C experiment=..."

print(config)

exp = config["experiment"]
strat = config["strategy"]

params = config["experiments"][exp]


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
            "logs/{exp}/{dset}_{family}/{strat}_{bb}_{arch}/{seed}/srf_estimates.csv",
            exp=[exp],
            family=params["families"],
            seed=list(range(params["num_seeds"])),
            dset=params["datasets"],
            arch=params["architectures"],
            strat=[strat],
            bb=params["backbones"],
        ),


rule impl:
    output:
        "logs/{exp}/{dset}_{family}/{strat}_{bb}_{arch}/{seed}/srf_estimates.csv",
    log:
        err="logs/{exp}/{dset}_{family}/{strat}_{bb}_{arch}/{seed}/err.log",
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
        " logdir=logs/{wildcards.exp}"
        ' subdir=""'
        " loggers.tb=false"
        " training.progbar=false"
        " &2> {log.err}"
        " && wait"
        # ^ note! without explicit 'wait', it exits early and fails
