import numpy as np
import torch
import argparse
from collections import defaultdict, deque
import pandas as pd
import os

from models.DensityNet import DensityNet
from dataset.dataset import get_iter, make_dataset, DATASETS, set_seed


def main(args: argparse.Namespace) -> None:
    set_seed(1234 + 131 * args.seed)  # like torch manual seed at all levels
    dev = torch.cuda if torch.cuda.is_available() else "cpu"
    delta_list = torch.linspace(0.5 / 20, 0.5, steps=20)
    D = make_dataset(args.dataset, delta_list, n_train=args.n_train, n_test=args.n_test)
    train_matrix = D["train_matrix"]
    input_dim = train_matrix.shape[1] - 2
    train_loader = get_iter(train_matrix, batch_size=500, shuffle=False)

    density_estimator_config = [(input_dim, 50, 1), (50, 50, 1)]
    num_grid = 10

    # define model
    model = DensityNet(density_estimator_config, num_grid).to(dev)
    model._initialize_weights()

    # Optimizer
    if args.opt == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
            weight_decay=args.wd,
        )
    elif args.opt == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.0001,
            momentum=0.9,
            weight_decay=args.wd,
            nesterov=True
        )

    for epoch in range(args.n_epoch):

        losses = defaultdict(lambda: deque(maxlen=len(train_loader)))
        for idx, item in enumerate(train_loader):
            t = item["treatment"].to(dev)
            x = item["covariates"].to(dev)
            y = item["outcome"].to(dev)

            optimizer.zero_grad()
            model_output = model.forward(t, x)
            probs = model_output["prob_score"]
            z = model_output["z"]

            # 1. negative loglikelihood loss
            total_loss = -probs.log().mean()

            total_loss.backward()
            optimizer.step()

            losses["total_loss"].append(total_loss.item())

        if epoch == 0 or (epoch + 1) % 100 == 0:
            if not args.silent:
                print("== Epoch: ", epoch, " ==")
                print("Metrics:")
                for k, vec in losses.items():
                    print(f"  {k}: {np.mean(vec):.4f}")

            # iptw estimates
            df = pd.DataFrame({"delta": delta_list})
            for part in ("train", "test"):
                M = D[part + "_matrix"].to(dev)
                t, x, y = M[:, 0], M[:, 1:-1], M[:, -1]
                z = model.forward(t, x)["z"]
                srf = D["srf_" + part]
                df[part + "_truth"] = srf

                estims = {}
                errors = {}
                for d, truth in zip(delta_list, srf):
                    if D["delta_transform"] == "percent":
                        shifted_t = t * (1 - d)
                        jacobian = (1 - d)
                    elif D["delta_transform"] == "additive":
                        scale = D["delta_scale"]
                        shifted_t = t - scale * d
                        jacobian = 1.0
                    r = model.density_estimator.density_ratio(
                        t, shifted_t, z, jacobian=jacobian, log=False
                    )
                    estim = (r / r.mean() * y).mean().item()
                    error = (estim - truth).pow(2).item()
                    key = np.round(d.item(), 2)
                    estims[key] = estim
                    errors[key] = error
                df[part + "_estim"] = estims.values()
                df[part + "_error"] = errors.values()
                av_curve_error = np.mean(list(errors.values()))

                if not args.silent:
                    print(f"SRF {part}:")
                    print(f"  @0.1: {errors[0.1]:.4f}")
                    print(f"  @0.5: {errors[0.5]:.4f}")
                    print(f"  curve: {av_curve_error:.4f}")

            results_path = f"results/{args.dataset}/{args.seed:02d}/out.csv"
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            df.round(4).to_csv(results_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--dataset", default="sim-N", type=str, choices=DATASETS)
    parser.add_argument("--opt", default="adam", type=str, choices=('adam', 'sgd'))
    parser.add_argument("--n_train", default=500, type=int)
    parser.add_argument("--n_test", default=200, type=int)
    parser.add_argument("--n_epoch", default=800, type=int)
    parser.add_argument("--wd", default=5e-3, type=int)
    parser.add_argument("--silent", default=False, action="store_true")
    

    args = parser.parse_args()
    main(args)
