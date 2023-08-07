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
        "logs/exp1/gaussian/{seed}/{dset}_{arch}_{strat}/srf_estimates.csv",
    log:
        err="logs/exp1/gaussian/{seed}/{dset}_{arch}_{strat}/stderr.log",
    shell:
        """
        python main.py dataset={wildcards.dset} architecture={wildcards.arch}\
          strategy={wildcards.strat} seed={wildcards.seed} logdir=logs/exp1\
          subdir="" &2> {log.err}
        """
