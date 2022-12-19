import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cat, stack, einsum, Tensor
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
import argparse
from minimal_ihdp import ExposureShiftCIHDP


class VCLinear(nn.Module):
    """Implements a varying coefficient linear layer"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        deg: int,
        knots: int,
        bias: bool = True,
        condition_dim: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.condition_dim = condition_dim
        self.deg = deg
        self.knots = knots
        self.num_basis = deg + 1 + len(knots)
        self.coef_dim = condition_dim * self.num_basis
        self.weight = nn.Parameter(torch.empty((self.coef_dim, input_dim, output_dim)))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_dim))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # identical to reset_parameters in nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def basis(self, condition: Tensor) -> Tensor:
        out = []
        for j in range(self.condition_dim):
            t = condition[:, j]
            for i in range(self.num_basis):
                if i <= self.deg:
                    out.append(t.pow(i))
                else:
                    out.append(F.relu(t - self.knots[i - self.deg - 1]).pow(self.deg))
        return stack(out, 1)

    def forward(self, x: Tensor) -> Tensor:
        m, c = x.shape[1], self.condition_dim
        x, condition = torch.split(x, [m - c, c], dim=1)
        basis = self.basis(condition)
        coefs = einsum("bc,cde->bde", basis, self.weight)
        out = einsum("bde,bd->be", coefs, x)
        if self.bias is not None:
            out = out + self.bias
        return out


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
        self.head = nn.Sequential(
            nn.Linear(input_dim, num_grid, bias=False), nn.LogSoftmax()
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        U = (t * self.N).ceil().long().clamp(max=self.N - 1)
        L = (U - 1).clamp(min=0)
        interp = 1 - (U - t * self.N)
        out = self.head(x)
        L_out = torch.gather(out, 1, L.unsqueeze(1)).squeeze(1)
        U_out = torch.gather(out, 1, U.unsqueeze(1)).squeeze(1)
        out = L_out + (U_out - L_out) * interp
        return out


class SIVCNet(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        out_deg: int,
        eps_deg: int,
        out_knots: list[float],
        eps_knots: list[float],
        dlist: list[float],
        hidden_dim: int = 50,
        act: str = "ReLU",
        num_hidden_layers_body: int = 1,
        num_hidden_layers_out: int = 1,
        num_density_grid: int = 20,
        targeted: bool = True,
        lr: float = 0.001,
        **kwargs,
    ) -> None:
        super().__init__()
        super().save_hyperparameters()

        h = hidden_dim
        dlist = torch.FloatTensor(dlist)
        self.register_buffer("dlist", dlist, persistent=True)
        self.lr = lr
        act = getattr(nn, act)

        # covariate representation
        layers = [nn.Linear(input_dim, h)]
        for _ in range(num_hidden_layers_body):
            layers.extend([nn.Linear(h, h), act()])
        self.body = nn.Sequential(*layers)

        # outcome
        self.Q = nn.ModuleList([VCLinear(h, 1, out_deg, out_knots)])
        for _ in range(num_hidden_layers_out):
            l = nn.Sequential(VCLinear(h, h, out_deg, out_knots), act())
            self.Q.insert(0, l)

        # density head
        self.logdens = ConditionalLogDensity(h, num_density_grid)

        # epsilon for targeted regularization
        self.targeted = targeted
        if targeted:
            self.tr = VCLinear(1, 1, eps_deg, eps_knots)

    def evalQ(self, hidden: Tensor, t: Tensor) -> Tensor:
        out = hidden
        for m in self.Q:
            out = m(cat([hidden, t.unsqueeze(1)], 1))
        return out.squeeze(1)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        hidden = self.body(x)
        yhat = self.evalQ(hidden, t)
        if self.targeted:
            ix = torch.randint(len(self.dlist), size=(x.shape[0],), device=x.device)
            d = self.dlist[ix]
            eps = self.tr(d.unsqueeze(1)).squeeze(1)
            dr = self.density_ratio(hidden, t, d)
            ytilde = yhat + eps * dr
        else:
            ytilde = yhat
        return ytilde, yhat, hidden

    def counterfactuals(self, hidden: Tensor, t: Tensor) -> Tensor:
        cfs = []
        if self.targeted:
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

    def density_ratio(self, hidden: Tensor, t: Tensor, d: Tensor) -> Tensor:
        logdr = self.logdens(hidden, t / d) - self.logdens(hidden, t)
        density_ratio = logdr.clamp(-5.0, 5.0).exp()
        return density_ratio

    def training_step(self, batch: tuple[Tensor], _: int) -> Tensor:
        x, t, y, _ = batch

        ytilde, yhat, hidden = self.forward(x, t)

        # 1. mse_loss
        mse_loss = F.mse_loss(yhat, y)
        self.log("mse", float(mse_loss), on_epoch=True, on_step=False)

        # 2. generalized propensity score loss/density estimation
        ps_loss = -self.logdens(hidden, t).clamp(-5.0, 5.0).mean()
        self.log("ps", float(ps_loss), on_epoch=True, on_step=False)

        # 3. targeted_loss
        tr_loss = F.mse_loss(ytilde, y) if self.targeted else 0.0
        self.log("tr", float(tr_loss), on_epoch=True, on_step=False)

        total = mse_loss + ps_loss + tr_loss
        self.log("total", float(total), on_epoch=True, on_step=False)

        return total

    def validation_step(self, batch: tuple[Tensor], batch_index: int) -> Tensor:
        x, t, y, cf = batch

        ytilde, yhat, hidden = self.forward(x, t)

        # 1. mse_loss
        mse_loss = F.mse_loss(yhat, y)
        self.log("vmse", float(mse_loss), on_epoch=True, on_step=False)

        # 2. generalized propensity score loss/density estimation
        ps_loss = -self.logdens(hidden, t).clamp(-5.0, 5.0).mean()
        self.log("vps", float(ps_loss), on_epoch=True, on_step=False)

        # 3. targeted loss
        tr_loss = F.mse_loss(ytilde, y) if self.targeted else 0.0
        self.log("vtr", float(tr_loss), on_epoch=True, on_step=False)

        # 4. counterfactual loss
        cf_hat = self.counterfactuals(hidden, t)
        cf_loss = F.mse_loss(cf, cf_hat)
        esrf_error = F.mse_loss(cf_hat.mean(0), cf.mean(0))
        self.log("vcf", float(cf_loss), on_epoch=True, on_step=False)
        self.log("vesrf", float(esrf_error), on_epoch=True, on_step=False)

    def configure_optimizers(self):
        core_modules = [self.body, self.Q, self.logdens]
        core_params = nn.ModuleList(core_modules).parameters()
        groups = [dict(params=core_params, weight_decay=1e-3)]
        if self.targeted:
            groups.append(dict(params=self.tr.parameters(), weight_decay=0.0))
        optim = torch.optim.Adam(groups, lr=self.lr)
        return optim


def main(args: argparse.Namespace) -> None:
    pl.seed_everything(args.seed, workers=True)

    dlist = np.linspace(0.5, 1.0, 100).tolist()
    datamodule = ExposureShiftCIHDP(
        args.ihdp_path, dlist=dlist, num_workers=8, batch_size=32
    )

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        auto_lr_find=True,
        gradient_clip_val=10,
        max_epochs=2000,
        logger=TensorBoardLogger(save_dir=args.logdir, name="minimal"),
        enable_checkpointing=True,
        default_root_dir=args.logdir,
    )

    model = SIVCNet(
        input_dim=25,
        dlist=dlist,
        out_deg=2,
        out_knots=[0.2, 0.4, 0.6, 0.8],
        eps_deg=3,
        eps_knots=[0.6, 0.7, 0.8, 0.9],
        hidden_dim=50,
        targeted=args.targeted,
        seed=args.seed,
        num_density_grid=50
    )
    trainer.fit(model, datamodule)

    # save the exposure shift response function
    model.eval()
    x, t, y, cf = datamodule.val[:]  # test dataset
    with torch.no_grad():
        cf_hat = model.counterfactuals(model.body(x), t)
        estim = cf_hat.mean(0).numpy()
        truth = cf.mean(0).numpy()
        results = pd.DataFrame({"delta": dlist, "estim": estim, "truth": truth})
        results.to_csv(f"{trainer.logger.log_dir}/esrf.csv", index=False)

    # some basic plots about the generated date
    sns.displot(t.numpy())
    plt.savefig(f"{trainer.logger.log_dir}/tdist.png"); plt.close()
    sns.displot(y.numpy())
    plt.savefig(f"{trainer.logger.log_dir}/ydist.png"); plt.close()
    plt.plot(dlist, cf.mean(0).numpy())
    plt.xlabel("fraction of exposure shift")
    plt.ylabel("mean outcome")
    plt.savefig(f"{trainer.logger.log_dir}/true_esrf.png"); plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ihdp_path", type=str, default="dataset/ihdp/ihdp.csv")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--notr", dest="targeted", default=True, action="store_false")
    parser.add_argument("--logdir", type=str, default="logs")
    args = parser.parse_args()
    main(args)
