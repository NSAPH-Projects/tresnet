import logging
import yaml
from glob import glob
from tqdm import tqdm

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def benchmarks_from_dir(root_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load results from an experiment directory. It recursively searches
    for all `srf_estimates.csv` files and loads them into a DataFrame.
    It uses the corresponding `args.yaml` file to extract the experiment
    metadata.

    Returns:
        results: DataFrame with all results, for every seed
        metrics: DataFrame with metrics, aggregated over seed
                 grouped by dataset, glm_family and experiment
    """
    files = glob("**/srf_estimates.csv", root_dir=root_dir, recursive=True)
    logger.warning(f"Extracting {len(files)} files that look like {files[0]}")

    results = []

    for f in tqdm(files):
        comps = f.split("/")
        expid = "_".join([comps[0], comps[1], comps[2], comps[3]])

        # config file
        conf_file = f.replace("srf_estimates.csv", "args.yaml")
        with open(root_dir + "/" + conf_file, "r") as io:
            conf = yaml.load(io, Loader=yaml.BaseLoader)

        # read pandas
        df = pd.read_csv(root_dir + "/" + f)
        df["dataset"] = conf["dataset"]
        df["family"] = conf["glm_family"]
        df["seed"] = conf["seed"]
        df["experiment"] = conf["experiment"]
        df["expid"] = expid
        df["setup"] = conf["dataset"] + "_" + conf["glm_family"]
        df["f"] = f
        df["shiftid"] = np.arange(df.shape[0])

        results.append(df)

    results = pd.concat(results)

    # obtain mse, variance, bias2 metrics
    error_metrics = results.groupby(
        ["dataset", "family", "experiment", "shiftid"]
    ).apply(_compute_metrics)
    error_metrics = (
        error_metrics.reset_index().groupby(["dataset", "family", "experiment"]).mean()
    ).drop(columns=["shiftid"])

    # rank by seed too. Ranks are compute from the average mse along the curve
    curve_train_mse = (
        results.groupby(["dataset", "family", "experiment", "seed"])
        .apply(lambda x: (x["train_srf"] - x["true_train_srf"]).mean() ** 2)
        .reset_index()
        .rename(columns={0: "train_mse"})
    )
    curve_test_mse = (
        results.groupby(["dataset", "family", "experiment", "seed"])
        .apply(lambda x: (x["test_srf"] - x["true_test_srf"]).mean() ** 2)
        .reset_index()
        .rename(columns={0: "test_mse"})
    )
    curve_train_mse["train_rank"] = (
        curve_train_mse.groupby(["dataset", "family", "seed"])
        .rank(ascending=False)
        .train_mse
    )
    curve_test_mse["test_rank"] = (
        curve_test_mse.groupby(["dataset", "family", "seed"])
        .rank(ascending=False)
        .test_mse
    )
    ranks = pd.concat([curve_train_mse, curve_test_mse], axis=1)

    # compute ROC from rankings
    roc_train = curve_train_mse.groupby(
        ["dataset", "family", "experiment"]
    ).train_rank.apply(np.cumsum)
    roc_train = roc_train.reset_index().rename(columns={"level_3": "curve_index"})
    roc_train["train_rank_norm"] = (
        roc_train.groupby(["dataset", "family"])
        .train_rank.apply(lambda x: x / x.max())
        .values
    )

    roc_test = curve_test_mse.groupby(
        ["dataset", "family", "experiment"]
    ).test_rank.apply(np.cumsum)
    roc_test = roc_test.reset_index().rename(columns={"level_3": "curve_index"})
    roc_test["test_rank_norm"] = (
        roc_test.groupby(["dataset", "family"])
        .test_rank.apply(lambda x: x / x.max())
        .values
    )

    roc = roc_train.merge(
        roc_test, on=["dataset", "family", "experiment", "curve_index"]
    )

    # auc from ROC
    auc = (
        roc.groupby(["dataset", "family", "experiment"])
        .mean()
        .drop(columns=["curve_index"])
    )

    return dict(
        results=results,
        metrics=error_metrics,
        ranks=ranks,
        roc=roc,
        auc=auc,
    )


def _compute_metrics(x: pd.DataFrame) -> pd.Series:
    out = dict(
        train_mse=(x["train_srf"] - x["true_train_srf"]).mean() ** 2,
        test_mse=(x["test_srf"] - x["true_test_srf"]).mean() ** 2,
        train_bias2=(x["train_srf"].mean() - x["true_train_srf"].mean()) ** 2,
        test_bias2=(x["test_srf"].mean() - x["true_test_srf"].mean()) ** 2,
        train_variance=x["train_srf"].var(),
        test_variance=x["test_srf"].var(),
    )
    return pd.Series(out.values(), index=out.keys())
