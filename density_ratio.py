import argparse
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cat, stack, einsum, Tensor
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
from minimal_ihdp import IHDP_C



class ConditionalLogDensity(nn.Module):
    def __init__(self, input_dim: int, num_grid: int) -> None:
        super().__init__()
        """
        Assume the variable is bounded by [0,1]
        the output grid: 0, 1/B, 2/B, ..., B/B; output dim = B + 1; num_grid = B
        """
        self.input_dim = input_dim
        self.N = num_grid
        self.outd = num_grid + 1
        self.head = nn.Linear(input_dim, num_grid, bias=False)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        N = self.N
        tN = t * N
        U = tN.ceil().long().clamp(0, N - 1)
        L = tN.floor().long().clamp(0, N - 1)
        interp = (tN - L) / N
        out = self.head(x)
        out = F.log_softmax(out, dim=1)
        L_out = torch.gather(out, 1, L.unsqueeze(1)).squeeze(1)
        U_out = torch.gather(out, 1, U.unsqueeze(1)).squeeze(1)
        out = L_out + (U_out - L_out) * interp
        return out


class DR(pl.LightningModule):
    def __init__(
        self,
        mode: str,
        input_dim: int,
        dlist: list[float],
        tlist: list[float],
        hidden_dim: int = 50,
        act: str = "ReLU",
        num_hidden_layers_body: int = 1,
        num_hidden_layers_out: int = 1,
        num_density_grid: int = 20,
        lr: float = 0.001,
        weight_decay: float = 5e-3,
        **kwargs,
    ) -> None:
        super().__init__()
        super().save_hyperparameters()

        h = hidden_dim
        dlist = torch.FloatTensor(dlist)
        tlist = torch.FloatTensor(tlist)
        self.register_buffer("dlist", dlist, persistent=True)
        self.register_buffer("tlist", tlist, persistent=True)
        act = getattr(nn, act)

        # covariate representation
        layers = [nn.Linear(input_dim, h)]
        for _ in range(num_hidden_layers_body):
            layers.extend([nn.Linear(h, h), act()])
        self.body = nn.Sequential(*layers)

        # density head
        if mode == "density":
            self.logdens = ConditionalLogDensity(h, num_density_grid)
        elif mode == "classifier":
            self.cl = nn.Linear(h, len(dlist))


    def evalQ(self, hidden: Tensor, t: Tensor) -> Tensor:
        out = hidden
        for m in self.Q:
            out = m(cat([hidden, t.unsqueeze(1)], 1))
        return out.squeeze(1)

    def forward(self, x: Tensor, t: Tensor, target: Optional[str] = None) -> Tensor:
        hidden = self.body(x)
        yhat = self.evalQ(hidden, t)
        if target is None:
            target = self.target
        if target == "es":
            ix = torch.randint(len(self.dlist), size=(x.shape[0],), device=x.device)
            d = self.dlist[ix]
            eps = self.tr(d.unsqueeze(1)).squeeze(1)
            dr = self.density_ratio(hidden, t, d, target=target)
            ytilde = yhat + eps * dr.detach()
        elif target == "dose":
            dr = self.density_ratio(hidden, t, t, target=target)
            eps = self.tr(t.unsqueeze(1)).squeeze(1)
            ytilde = yhat + eps * dr.detach()
        else:
            ytilde = yhat.detach()
        return ytilde, yhat, hidden

    def es_counterfactuals(self, hidden: Tensor, t: Tensor) -> Tensor:
        cfs = []
        if self.target == "es":
            eps = self.tr(self.dlist.unsqueeze(1)).squeeze(1)
            for eps_d, d in zip(eps, self.dlist.squeeze()):
                ycf = self.evalQ(hidden, t * d)
                dr = self.density_ratio(hidden, t, d)
                ycf = ycf + eps_d * dr
                cfs.append(ycf)
        else:
            for d in self.dlist.squeeze():
                ycf = self.evalQ(hidden, t * d)
                cfs.append(ycf)
        return stack(cfs, 1)

    def dose_counterfactuals(self, hidden: Tensor, t: Tensor) -> Tensor:
        cfs = []
        if self.target == "dose":
            eps = self.tr(self.tlist.unsqueeze(1)).squeeze(1)
            for eps_t, ti in zip(eps, self.tlist.squeeze()):
                ti = torch.full_like(t, ti)
                ycf = self.evalQ(hidden, ti)
                dr = self.density_ratio(hidden, ti, ti)
                ycf = ycf + eps_t * dr
                cfs.append(ycf)
        else:
            for ti in self.tlist.squeeze():
                ycf = self.evalQ(hidden, torch.full_like(t, ti))
                cfs.append(ycf)
        return stack(cfs, 1)

    def density_ratio(
        self, hidden: Tensor, t: Tensor, factor: Tensor, target: Optional[str] = None
    ) -> Tensor:
        if target is None:
            target = self.target
        if target == "es":
            logdr = (
                factor.log()
                + self.logdens(hidden, t / factor)
                - self.logdens(hidden, t)
            )
        elif target == "dose":
            logdr = -self.logdens(hidden, t)
        density_ratio = logdr.clamp(max=7.0).exp()
        return density_ratio

    def training_step(self, batch: tuple[Tensor], _: int) -> Tensor:
        x, t, y, *_ = batch

        ytilde, yhat, hidden = self.forward(x, t)

        # 1. mse_loss
        mse_loss = F.mse_loss(yhat, y)
        self.log("mse", float(mse_loss), on_epoch=True, on_step=False)

        # 2. generalized propensity score loss/density estimation
        negll = -self.logdens(hidden, t)
        ps_loss = negll.mean()
        self.log("ps", float(ps_loss), on_epoch=True, on_step=False)

        # 3. target_loss
        tr_loss = F.mse_loss(ytilde, y) if self.target else 0.0
        self.log("tr", float(tr_loss), on_epoch=True, on_step=False)

        total = mse_loss + ps_loss + tr_loss
        self.log("total", float(total), on_epoch=True, on_step=False)

        return total

    def validation_step(self, batch: tuple[Tensor], batch_index: int) -> Tensor:
        x, t, y, cf_es, cf_dose = batch

        ytilde, yhat, hidden = self.forward(x, t)

        # 1. mse_loss
        mse_loss = F.mse_loss(yhat, y)
        self.log("vmse", float(mse_loss), on_epoch=True, on_step=False)

        # 2. generalized propensity score loss/density estimation
        ps_loss = -self.logdens(hidden, t).mean()
        self.log("vps", float(ps_loss), on_epoch=True, on_step=False)

        # 3. target loss
        tr_loss = F.mse_loss(ytilde, y) if self.target else 0.0
        self.log("vtr", float(tr_loss), on_epoch=True, on_step=False)

        # 4. exposure shift counterfactual loss
        cf_hat = self.es_counterfactuals(hidden, t)
        cf_loss = F.mse_loss(cf_es, cf_hat)
        error = F.mse_loss(cf_es.mean(0), cf_hat.mean(0))
        self.log("vcf_es", float(cf_loss), on_epoch=True, on_step=False)
        self.log("vrc_es", float(error), on_epoch=True, on_step=False)

        # 5. average curve counterfactual loss
        cf_hat = self.dose_counterfactuals(hidden, t)
        cf_loss = F.mse_loss(cf_dose, cf_hat)
        error = F.mse_loss(cf_dose.mean(0), cf_hat.mean(0))
        self.log("vcf_dose", float(cf_loss), on_epoch=True, on_step=False)
        self.log("vrc_dose", float(error), on_epoch=True, on_step=False)

    def configure_optimizers(self):
        core_modules = [self.body, self.Q, self.logdens]
        core_params = nn.ModuleList(core_modules).parameters()
        wd, lr = [self._hparams[k] for k in ("weight_decay", "lr")]
        groups = [dict(params=core_params, weight_decay=wd)]
        if self.target:
            groups.append(dict(params=self.tr.parameters(), weight_decay=0.0))
        optim = torch.optim.Adam(groups, lr=lr)
        return optim


def main(args: argparse.Namespace) -> None:
    pl.seed_everything(args.seed, workers=True)

    dlist = np.arange(0.05, 1.0, 0.05).tolist()
    tlist = np.arange(0.05, 1.0, 0.05).tolist()
    datamodule = IHDP_C(
        args.ihdp_path, dlist=dlist, tlist=tlist, num_workers=8, batch_size=32
    )

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        auto_lr_find=True,
        gradient_clip_val=100,
        max_epochs=5000,
        logger=TensorBoardLogger(save_dir=args.logdir, name="minimal"),
        enable_checkpointing=False,
        default_root_dir=args.logdir,
    )

    model = SIVCNet(
        input_dim=25,
        dlist=dlist,
        tlist=tlist,
        deg_Q=2,
        deg_eps=2,
        knots_Q=[0.33, 0.66],
        knots_eps=np.arange(0.1, 1, 0.1).tolist(),
        hidden_dim=50,
        num_hidden_layers_body=0,
        num_hidden_layers_out=0,
        act="ReLU",
        num_density_grid=20,
        target=args.target,
        seed=args.seed,
        weight_decay=args.weight_decay,
    )
    trainer.fit(model, datamodule)

    # save the exposure shift response function
    model.eval()
    x, t, y, cf_es, cf_dose = datamodule.val[:]  # test dataset
    with torch.no_grad():
        hidden = model.body(x)
        cf_hat = model.es_counterfactuals(hidden, t)
        estim = cf_hat.mean(0).numpy()
        truth = cf_es.mean(0).numpy()
        results = pd.DataFrame({"delta": dlist, "estim": estim, "truth": truth})
        results.to_csv(f"{trainer.logger.log_dir}/esrf.csv", index=False)

        cf_hat = model.dose_counterfactuals(hidden, t)
        estim = cf_hat.mean(0).numpy()
        truth = cf_dose.mean(0).numpy()
        results = pd.DataFrame({"t": tlist, "estim": estim, "truth": truth})
        results.to_csv(f"{trainer.logger.log_dir}/adrf.csv", index=False)

    # some basic plots about the generated date
    sns.displot(t.numpy())
    plt.savefig(f"{trainer.logger.log_dir}/tdist.png")
    plt.close()
    sns.displot(y.numpy())
    plt.savefig(f"{trainer.logger.log_dir}/ydist.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ihdp_path", type=str, default="dataset/ihdp/ihdp.csv")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--target", default="dose", type=str, choices=("dose", "es", "none")
    )
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    args = parser.parse_args()
    main(args)
