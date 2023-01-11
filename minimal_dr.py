import argparse
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Distribution
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import random_split, DataLoader, TensorDataset
import torch.distributions as td
from pyro.distributions import AsymmetricLaplace


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
        in_supp = ((0.0 <= t) & (t <= 1.0)).long()
        U = tN.ceil().long().clamp(0, N - 1)
        L = tN.floor().long().clamp(0, N - 1)
        interp = (tN - L) / N
        out = self.head(x)
        out = F.log_softmax(out, dim=1)
        L_out = torch.gather(out, 1, L.unsqueeze(1)).squeeze(1)
        U_out = torch.gather(out, 1, U.unsqueeze(1)).squeeze(1)
        out = L_out + (U_out - L_out) * interp
        return out * in_supp


class DummyData(pl.LightningDataModule):

    def __init__(
        self,
        dlist: list[float],
        batch_size: int = 32,
        num_workers: int = 0,
        N: int = 100
    ) -> None:
        super().__init__()
        self.dlist = dlist
        self.loader_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=True,
        )
        Z = (torch.rand(N) < 0.75).float() # aux latent
        noise_A = torch.randn(N)
        X = (Z * (-3 + 0.5 * noise_A) + (1 - Z) * (0.5 + 0.5 * noise_A)).sigmoid() # Urbanization indicator
        X = 0.25 + (X - X.min()) / (X.max() - X.min())
        dist = AsymmetricLaplace(X.sqrt(), 0.05, 0.5)
        A = dist.sample() # air pollution
        D = torch.FloatTensor(np.random.choice(dlist, size=N, replace=True))
        R = (dist.log_prob(A - D) - dist.log_prob(A)).exp()
        f = lambda X, A, noise: 100 * X * (0.25 + (10 * (2 * A - 0.75)).sigmoid()) + noise
        noise_Y = np.random.poisson(torch.rand(N))
        Y = f(X, A, noise_Y)
        Ycf = torch.stack([f(X, A + d, noise_Y) for d in dlist],1)
        self.full = TensorDataset(X[:, None], A, D, R, Y, Ycf)
        self.train, self.val = random_split(self.full, [0.5, 0.5])

    def train_dataloader(self):
        return DataLoader(self.train, **self.loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val, **self.loader_kwargs)


class DRatioNet(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        dlist: list[float],
        hidden_dim: int = 50,
        act: str = "ReLU",
        num_hidden_layers: int = 1,
        lr: float = 0.001,
        weight_decay: float = 5e-3,
        reg: bool = False,
        erm: bool = True,
        ngrid: int = 10,
        gam: float = 0.001,  # generally 1/N,
        train_data: Optional[TensorDataset] = None,
        val_data: Optional[TensorDataset] = None,
        fitreg: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        super().save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay
        self.reg = reg
        self.erm = erm
        self.fitreg = fitreg
        self.gam = gam
        if fitreg:
            self.lsig = nn.Parameter(torch.tensor(1.0))

        h = hidden_dim
        dlist = torch.FloatTensor(dlist)
        self.register_buffer("dlist", dlist)
        self.train_data = train_data
        self.val_data = val_data
        act = getattr(nn, act)

        # covariate representation
        layers = [nn.Linear(input_dim, h)]
        for _ in range(num_hidden_layers):
            layers.extend([nn.Linear(h, h), act()])
        self.body = nn.Sequential(*layers)

        # density head
        self.dens = ConditionalLogDensity(h, ngrid)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)

    def dratio_reg(self, hidden: Tensor, t: Tensor, d: Tensor) -> Tensor:
        # Z = torch.rand_like(t) < 0.5
        # ttilde = torch.where(Z, t + d, t)
        # logits = g.log_prob(ttilde - d) - g.log_prob(ttilde)
        # Z = torch.cat([torch.ones_like(t), torch.zeros_like(t)]).clamp(eps, 1.0 - eps)
        # logits1 = g.log_prob(t) - g.log_prob(t + d)
        # logits0 = g.log_prob(t - d) - g.log_prob(t)
        # logits = torch.cat([logits1, logits0])
        # return F.binary_cross_entropy_with_logits(logits, Z.float())

        r = self.dr(hidden, t, d)
        diffs = r.log().diff(n=2)
        tgt = torch.zeros_like(diffs)
        sig = self.lsig.clamp(max=5.0).exp()
        loss = (self.gam / (1e-3 + sig)**2) * F.huber_loss(diffs, tgt, delta=1.0)
        if self.fitreg:
            loss = loss - td.HalfCauchy(1.0).log_prob(sig)
        return loss, r

    def dr(self, hidden: Tensor, a: Tensor, d: Tensor, log=False) -> Tensor:
        log_g1 = self.dens(hidden, a - d)
        log_g0 = self.dens(hidden, a)
        out = log_g1 - log_g0
        if not log:
            out = out.clamp(max=10).exp()
        return out

    def irf_from_iptw(self, hidden: Tensor, a: Tensor, y: Tensor) -> Tensor:
        irfhat = []
        for d_ in self.dlist:
            d = torch.full_like(a, d_)
            r = self.dr(hidden, a, d)
            r = r / r.mean()
            irfhat.append((r * y).mean())
        return torch.stack(irfhat)

    def training_step(self, batch: tuple[Tensor], _: int) -> Tensor:
        x, a, d, r, *_ = batch

        hidden = self.forward(x)
        loss = torch.tensor(0.0).to(x.device)

        # 1. erm
        erm_loss = - self.dens(hidden, a).mean()
        if self.erm:
            loss = loss + erm_loss

        # 2. density ratio
        reg_loss, rhat = self.dratio_reg(hidden, a, d)
        if self.reg:
            loss = loss + reg_loss
            sig = self.lsig.clamp(max=5.0).exp() 
            self.log("sig", sig.item(), on_epoch=True, on_step=False)

        # 3. true dratio
        r_loss = F.mse_loss(rhat, r)

        self.log("erm", erm_loss.item(), on_epoch=True, on_step=False)
        self.log("reg", reg_loss.item(), on_epoch=True, on_step=False)
        self.log("dr", r_loss.item(), on_epoch=True, on_step=False)

        return loss
    
    def on_train_epoch_end(self) -> None:
        if not self.train_data:
            return
        X, A, _, _, Y, Ycf = self.train_data.dataset.tensors
        hidden = self.body(X)
        irfhat = self.irf_from_iptw(hidden, A, Y)
        irf = Ycf.mean(0)
        irf_error = F.mse_loss(irf, irfhat)
        self.log("irf", irf_error.item())


    def validation_step(self, batch: tuple[Tensor], batch_index: int) -> Tensor:
        x, a, d, r, *_ = batch

        hidden = self.forward(x)
        loss = torch.tensor(0.0).to(x.device)

        # 1. erm
        erm_loss = - self.dens(hidden, a).mean()
        if self.erm:
            loss = loss + erm_loss

        # 2. density ratio
        reg_loss, rhat = self.dratio_reg(hidden, a, d)
        if self.reg:
            loss = loss + reg_loss

        # 3. true dratio
        r_loss = F.mse_loss(rhat, r)

        self.log("verm", erm_loss.item(), on_epoch=True, on_step=False)
        self.log("vreg", reg_loss.item(), on_epoch=True, on_step=False)
        self.log("vdr", r_loss.item(), on_epoch=True, on_step=False)

        return loss

    def on_validation_epoch_end(self) -> None:
        if not self.val_data:
            return
        X, A, _, _, Y, Ycf = self.val_data.dataset.tensors
        hidden = self.body(X)
        irfhat = self.irf_from_iptw(hidden, A, Y)
        irf = Ycf.mean(0)
        irf_error = F.mse_loss(irf, irfhat)
        self.log("virf", irf_error.item())

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # sched = torch.optim.lr_scheduler.StepLR(optim, 5000, gamma=0.1)
        # return dict(optimizer=optim, lr_scheduler=sched)
        return optim


def main(args: argparse.Namespace) -> None:
    pl.seed_everything(args.seed, workers=True)

    dlist = np.arange(-0.2, 0, 0.01).tolist()
    data = DummyData(
        N=args.N, dlist=dlist, num_workers=8, batch_size=32
    )

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        auto_lr_find=False,
        gradient_clip_val=100,
        max_epochs=20000,
        logger=TensorBoardLogger(save_dir=args.logdir, name="dr"),
        enable_checkpointing=False,
        default_root_dir=args.logdir,
    )

    model = DRatioNet(
        input_dim=1,
        dlist=dlist,
        hidden_dim=16,
        num_hidden_layers=1,
        act="ELU",
        seed=args.seed,
        weight_decay=args.weight_decay,
        erm=args.erm,
        reg=args.reg,
        lr=0.001,
        gam=args.gam if args.gam else 1/args.N,
        train_data=data.train,
        val_data=data.val,
        fitreg=args.fitreg
    )
    trainer.fit(model, data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--no_reg", default=True, dest="reg", action="store_false")
    parser.add_argument("--no_erm", default=True, dest="erm", action="store_false")
    parser.add_argument("--no_fitreg", default=True, dest="fitreg", action="store_false")
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--gam", type=float, default=None)
    args = parser.parse_args()
    main(args)
