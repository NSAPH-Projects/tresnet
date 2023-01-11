import argparse
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
from typing import Optional
import matplotlib.pyplot as plt


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
        exo: float = 0.1,
        N: int = 100
    ) -> None:
        super().__init__()
        self.dlist = dlist
        self.loader_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            pin_memory=True,
        )
        Z = (torch.rand(N) < 0.75).float() # aux latent
        noise_A = torch.randn(N)
        Z = (torch.rand(N) < 0.75).float() # aux latent
        noise_A = torch.randn(N)
        X = (Z * (-3 + 0.5 * noise_A) + (1 - Z) * (0.25 + 0.5 * noise_A)).sigmoid() # Urbanization indicator
        skew = 0.5
        dist = AsymmetricLaplace(X.sqrt(), exo, skew)
        A = dist.sample() # air pollution
        D = torch.FloatTensor(np.random.choice(dlist, size=N, replace=True))
        R = (dist.log_prob(A - D) - dist.log_prob(A)).exp()
        f = lambda X, A, noise: 100 * X * (0.25 + (10 * (A - 1)).sigmoid()) + noise
        noise_Y = np.random.poisson(torch.rand(N))
        Y = f(X, A, noise_Y)
        Ycf = torch.stack([f(X, A + d, noise_Y) for d in dlist], 1)
        Rcf = torch.stack([(dist.log_prob(A - d) - dist.log_prob(A)).exp() for d in dlist], 1)
        self.full = TensorDataset(X[:, None], A, D, R, Y, Ycf, Rcf)
        self.train, self.val = random_split(self.full, [0.5, 0.5])

    def train_dataloader(self):
        return DataLoader(self.train, **self.loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val, **self.loader_kwargs)


class DRatioNet(pl.LightningModule):
    def __init__(
        self,
        destim: str,
        reg: str,
        input_dim: int,
        dlist: list[float],
        hidden_dim: int = 50,
        act: str = "ReLU",
        num_hidden_layers: int = 1,
        lr: float = 0.001,
        weight_decay: float = 5e-3,
        erm: bool = True,
        ngrid: int = 10,
        gam: float = 0.001,  # generally 1/N,
        train_data: Optional[TensorDataset] = None,
        val_data: Optional[TensorDataset] = None,
        fitreg: bool = True,
        fl_ord: int = 1,
        dropout: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__()
        super().save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay
        self.reg = reg
        self.destim = destim
        self.erm = erm
        self.fitreg = fitreg
        self.gam = gam
        self.fl_ord = fl_ord
        self.lsig = nn.Parameter(torch.tensor(1.0), requires_grad=fitreg)

        if reg == "flx":
            self.lsig2 = nn.Parameter(torch.tensor(1.0))
            self.h = nn.Parameter(torch.tensor(0.0))
        if reg == "var":
            self.lsig2 = nn.Parameter(torch.tensor(1.0))


        h = hidden_dim
        self.dlist = dlist
        self.train_data = train_data
        self.val_data = val_data
        act = getattr(nn, act)

        # covariate representation
        layers = [nn.Linear(input_dim, h), act(), nn.Dropout(dropout)]
        for _ in range(num_hidden_layers):
            layers.extend([nn.Linear(h, h), act()])
        self.body = nn.Sequential(*layers)

        # density head
        if destim == "discrete":
            self.dens = ConditionalLogDensity(h, ngrid)
        elif destim == "mix1":
            self.dparams = nn.Parameter(torch.randn(2 * ngrid))
            self.dwts = nn.Linear(h, ngrid, bias=False)
        elif destim == "mix2":
            self.dparams = nn.Linear(h, 2 * ngrid)
            self.dwts = nn.Parameter(torch.randn(ngrid))


    def dratio_reg(self, v: Tensor | Distribution, a: Tensor, d: Tensor, x: Tensor, eps: float=0.0) -> Tensor:
        if self.reg == "ratio1":
            Z = torch.rand_like(a) < 0.5
            a1 = torch.where(Z, a + d, a)
            logr = self.dr(v, a1, d, log=True)
            tgt = Z.float().clamp(eps, 1.0 - eps)
            loss = F.binary_cross_entropy_with_logits(logr, tgt)
        elif self.reg == "ratio2":
            Z = torch.cat([torch.ones_like(a), torch.zeros_like(a)]).clamp(eps, 1.0 - eps)
            logits1 = self.dr(v, a + d, d, log=True)
            logits0 = self.dr(v, a, d, log=True)
            logr = torch.cat([logits1, logits0])
            tgt = Z.float().clamp(eps, 1.0 - eps)
            loss = F.binary_cross_entropy_with_logits(logr, tgt)
        elif self.reg == "fl":
            dlist = self.dlist if self.dlist[-1] == 0.0 else self.dlist + [0.0]
            logr = torch.stack([self.dr(v, a, d_, log=True) for d_ in dlist], 1)
            diffs = logr.diff(n=self.fl_ord + 1)
            tgt = torch.zeros_like(diffs)
            sig = 1e-3 + self.lsig.clamp(max=5.0).exp()
            # loss = C * F.huber_loss(diffs, tgt, delta=1.0, reduction='none').sum(-1).mean()
            loss = - self.gam * td.Normal(0.0, sig).log_prob(diffs).sum(-1).mean()
            if self.fitreg:
                loss = loss - td.HalfCauchy(1.0).log_prob(sig)
        elif self.reg == "var":
            # logr = self.dr(v, a, d, log=True)
            # diffs = logr.clamp(-10.0, 10.0).diff(n=self.fl_ord + 1)
            # tgt = torch.zeros_like(diffs)
            # sig = self.lsig.clamp(max=5.0).exp()
            # C = (self.gam / (1e-3 + sig)**2)
            # loss = C * F.huber_loss(diffs, tgt, delta=1.0, reduction='none').sum(-1).mean()
            logr = torch.stack([self.dr(v, a, d_, log=True) for d_ in self.dlist], 1)
            sig = 1e-3 + self.lsig.clamp(max=5.0).exp()
            # sig2 = 1e-3 + self.lsig2.clamp(max=5.0).exp()
            # C = (self.gam / (1e-3 + sig)**2)
            # loss = C * logr.var(0).sum()
            loss = - self.gam * td.Normal(0, sig).log_prob(logr - logr.mean(0)).sum(-1).mean()
            # loss = loss + self.gam * sig2 * logr.mean(0).pow(2).sum()
            if self.fitreg:
                loss = loss - td.HalfCauchy(1.0).log_prob(sig)
                # loss = loss - td.HalfCauchy(1.0).log_prob(sig2)
        elif self.reg == "var2":
            logr = torch.stack([self.dr(v, a, d_, log=True) for d_ in self.dlist], 1)
            sig = 1e-3 + self.lsig.clamp(max=5.0).exp()
            loss = self.gam * sig * logr.mean(0).pow(2).sum()
            if self.fitreg:
                loss = loss - td.HalfCauchy(1.0).log_prob(sig)
        elif self.reg == "flx":
            logr = self.dr(v, a, d, log=True)
            diffs = logr.clamp(-10.0, 10.0).diff(n=2)
            tgt = torch.zeros_like(diffs)
            sig = self.lsig.clamp(max=5.0).exp()
            C = (self.gam / (1e-3 + sig)**2)
            loss1 = C * F.huber_loss(diffs, tgt, delta=1.0, reduction='none').sum(-1).mean()

            diffs2 = logr.clamp(-10.0, 10.0).diff(n=1)
            sig2 = self.lsig2.clamp(max=5.0).exp()
            C = (self.gam / (1e-3 + sig2)**2)
            tgt = torch.zeros_like(diffs2)
            loss2 = C * F.huber_loss(diffs2, tgt, delta=1.0, reduction='none').sum(-1).mean()

            w = self.h.sigmoid()
            loss = w * loss1 + (1 - w) * loss2
    
            if self.fitreg:
                loss = loss - td.HalfCauchy(1.0).log_prob(sig)
                loss = loss - td.HalfCauchy(1.0).log_prob(sig2)

        elif self.reg == "fl+ratio2":
            dlist = self.dlist if self.dlist[-1] == 0.0 else self.dlist + [0.0]
            logr = torch.stack([self.dr(v, a, d_, log=True) for d_ in dlist], 1)
            diffs = logr.diff(n=self.fl_ord + 1)
            tgt = torch.zeros_like(diffs)
            sig = 1e-3 + self.lsig.clamp(max=5.0).exp()
            # loss = C * F.huber_loss(diffs, tgt, delta=1.0, reduction='none').sum(-1).mean()
            loss = - self.gam * td.Normal(0.0, sig).log_prob(diffs).sum(-1).mean()
            if self.fitreg:
                loss = loss - td.HalfCauchy(1.0).log_prob(sig)
            #
            Z = torch.cat([torch.ones_like(a), torch.zeros_like(a)]).clamp(eps, 1.0 - eps)
            logits1 = self.dr(v, a + d, d, log=True)
            logits0 = self.dr(v, a, d, log=True)
            logr = torch.cat([logits1, logits0])
            tgt = Z.float().clamp(eps, 1.0 - eps)
            loss = loss + F.binary_cross_entropy_with_logits(logr, tgt)

        elif self.reg == "fl_gonly":
            if self.destim.startswith("mix"):
                log_g = v.log_prob(a)
            elif self.destim == "discrete":
                log_g = self.dens(v, a)

        elif self.reg == "none":
            loss = torch.tensor(0.0).to(a.device)
        elif self.reg == "post":
            dlist = self.dlist if self.dlist[-1] == 0.0 else self.dlist + [0.0]
            logr = torch.stack([self.dr(v, a, d_, log=True) for d_ in dlist], 1)
            v1 = self(x)
            logr1 = torch.stack([self.dr(v1, a, d_, log=True) for d_ in dlist], 1)
            sig = 1e-3 + self.lsig.clamp(max=5.0).exp()
            loss = - self.gam * td.Normal(0.0,  sig).log_prob(logr1 - logr).sum(1).mean()
        else:
            raise NotImplementedError
        return loss

    def make_dist(self, hidden: Tensor):
        if self.destim == "mix1":
            mu, lsig = torch.chunk(self.dparams, chunks=2)
            sig = lsig.clamp(max=5).exp()
            dist = td.Normal(mu, sig).expand((hidden.shape[0], len(mu)))
            wts = td.Categorical(logits=self.dwts(hidden))
            dist = td.MixtureSameFamily(wts, dist)
        elif self.destim == "mix2":
            mu, lsig = torch.chunk(self.dparams(hidden), chunks=2, dim=1)
            sig = lsig.clamp(max=5).exp()
            dist = td.Normal(mu, sig)
            wts = td.Categorical(logits=self.dwts)
            dist = td.MixtureSameFamily(wts, dist)
        return dist

    def forward(self, x: Tensor) -> (Tensor | Distribution):
        x = self.body(x)
        if self.destim.startswith("mix"):
            return self.make_dist(x)
        else:
            return x

    def dr(self,  v: Tensor | Distribution, a: Tensor, d: Tensor, log=False) -> Tensor:
        if self.destim.startswith("mix"):
            log_g1 = v.log_prob(a - d)
            log_g0 = v.log_prob(a)
        elif self.destim == "discrete":
            log_g1 = self.dens(v, a - d)
            log_g0 = self.dens(v, a)
        else:
            raise NotImplementedError
        out = log_g1 - log_g0
        if not log:
            out = out.clamp(max=10).exp()
        return out

    def irf_from_iptw(self, hidden: Tensor, a: Tensor, y: Tensor) -> Tensor:
        Rcf = []
        for d in self.dlist:
            r = self.dr(hidden, a, d, log=True)
            Rcf.append(r.exp())
        Rcf = torch.stack(Rcf, 1)
        irfhat = (Rcf / Rcf.mean(0, keepdims=True) * y[:, None]).mean(0)
        # irfhat = (Rcf * y[:, None]).mean(0)
        return irfhat

    def training_step(self, batch: tuple[Tensor], _: int) -> Tensor:
        x, a, d, r, *_ = batch

        v = self.forward(x)
        loss = torch.tensor(0.0).to(x.device)

        # 1. erm
        erm_loss = - self.log_prob(v, a).mean()
        if self.erm:
            loss = loss + erm_loss

        # 2. density ratio
        reg_loss = self.dratio_reg(v, a, d, x)
        rhat = self.dr(v, a, d)
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
        X, A, _, _, Y, Ycf, Rcf = self.train_data.dataset.tensors
        irfhat = self.irf_from_iptw(self(X), A, Y).detach()
        irfhat_iptw0 = (Rcf / Rcf.mean(0, keepdims=True) * Y[:, None]).mean(0)
        # irfhat_iptw0 = (Rcf * Y[:, None]).mean(0)
        irf = Ycf.mean(0)
        self.log("irf", F.mse_loss(irf, irfhat).item())
        self.log("irf_iptw0", F.mse_loss(irf, irfhat_iptw0).item())
        self.log("irf_hats", F.mse_loss(irfhat, irfhat_iptw0).item())

        epoch = self.current_epoch
        if epoch % 100 == 0:
            fig = plt.figure(figsize=(3, 2))
            plt.plot(irf, label="irf true", c="black")
            plt.plot(irfhat, label="irf_hat nn", c="red")
            plt.plot(irfhat_iptw0, label="irf_hat ps0", c="blue")
            plt.legend()
            self.logger.experiment.add_figure('generated_images', fig, epoch, close=True) 

    def log_prob(self, v: Tensor | Distribution, a: Tensor):
        if self.destim.startswith("mix"):
            return v.log_prob(a)
        elif self.destim == "discrete":
            return self.dens(v, a)
        else:
            raise NotImplementedError

    def validation_step(self, batch: tuple[Tensor], batch_index: int) -> Tensor:
        x, a, d, r, *_ = batch

        v = self.forward(x)
        loss = torch.tensor(0.0).to(x.device)

        # 1. erm
        erm_loss = - self.log_prob(v, a).mean()
        if self.erm:
            loss = loss + erm_loss

        # 2. density ratio
        reg_loss = self.dratio_reg(v, a, d, x)
        rhat = self.dr(v, a, d)
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
        X, A, _, _, Y, Ycf, Rcf = self.val_data.dataset.tensors
        irfhat = self.irf_from_iptw(self(X), A, Y).detach()
        irfhat_iptw0 = (Rcf / Rcf.mean(0, keepdims=True) * Y[:, None]).mean(0)
        # irfhat_iptw0 = (Rcf * Y[:, None]).mean(0)
        irf = Ycf.mean(0)
        self.log("virf", F.mse_loss(irf, irfhat).item())
        self.log("virf_iptw0", F.mse_loss(irf, irfhat_iptw0).item())
        self.log("virf_hats", F.mse_loss(irfhat, irfhat_iptw0).item())

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # sched = torch.optim.lr_scheduler.StepLR(optim, 5000, gamma=0.1)
        # return dict(optimizer=optim, lr_scheduler=sched)
        return optim


def main(args: argparse.Namespace) -> None:
    pl.seed_everything(args.seed, workers=True)

    dlist = np.linspace(-0.5, 0.0, 20).tolist()
    data = DummyData(
        N=args.N, dlist=dlist, num_workers=2, batch_size=32
    )

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        auto_lr_find=False,
        gradient_clip_val=100,
        max_epochs=20000,
        logger=TensorBoardLogger(save_dir=args.logdir, name="dr2"),
        enable_checkpointing=False,
        default_root_dir=args.logdir,
    )

    model = DRatioNet(
        destim=args.destim,
        reg=args.reg,
        input_dim=1,
        dlist=dlist,
        hidden_dim=16,
        num_hidden_layers=1,
        act="SiLU",
        seed=args.seed,
        weight_decay=args.weight_decay,
        erm=args.erm,
        lr=0.001,
        gam=args.gam if args.gam else 1/args.N,
        train_data=data.train,
        val_data=data.val,
        fitreg=args.fitreg,
        ngrid=args.ngrid,
        fl_ord=args.fl_ord,
        dropout=args.dropout,
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=10000)
    parser.add_argument("--fl_ord", type=int, default=1)
    parser.add_argument("--ngrid", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--reg", type=str, default="post", choices=("var", "var2", "fl", "ratio1", "ratio2", "fl+ratio2", "none", "flx", "post"))
    parser.add_argument("--destim", type=str, default="mix1", choices=("mix1", "mix2", "discrete"))
    parser.add_argument("--no_erm", default=True, dest="erm", action="store_false")
    parser.add_argument("--no_fitreg", default=True, dest="fitreg", action="store_false")
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--gam", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=0.25)
    args = parser.parse_args()
    main(args)
