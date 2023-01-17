import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists


def log_density_ratio_under_shift(
    treatment, delta, density_estimator, z, shift_type, eps=1e-10
):
    """z is the hidden vector for efficiency,
    eps just avoids nan on the log"""
    assert shift_type in ("subtract", "percent")
    if shift_type == "subtract":
        t1 = treatment + delta
        numer = density_estimator(t1, z)
    elif shift_type == "percent":
        t1 = treatment / (1 - delta)
        numer = (1 - delta) * density_estimator(t1, z)
    else:
        raise NotImplementedError(shift_type)
    denom = density_estimator(treatment, z)
    log_ratio = torch.log(eps + numer) - torch.log(eps + denom)

    return log_ratio


class RatioRegularizer(nn.Module):
    def __init__(self, delta_list, multiscale: bool = True, fit_scale: bool = True) -> None:
        super().__init__()
        self.register_buffer("multiscale", torch.tensor(multiscale))
        if multiscale:
            self.lsig = nn.Parameter(torch.zeros_like(delta_list), requires_grad=fit_scale)
        else:
            self.lsig = nn.Parameter(torch.tensor(0.0), requires_grad=fit_scale)
        self.register_buffer("delta_list", delta_list)

    def loss(self, treatment, density_estimator, z, shift_type):
        losses = []

        # sample delta at random for each element
        ix = torch.randint(
            high=len(self.delta_list),
            size=(treatment.shape[0], ),
            device=treatment.device
        )
        delta = self.delta_list[ix]

        # make a pseudo transformed treatment
        if shift_type == "subtract":
            shifted = treatment - delta
        elif shift_type == "percent":
            shifted = treatment * (1 - delta)

        # obtain density ratio logits for shited and normal
        logits_shifted = log_density_ratio_under_shift(
            shifted, delta, density_estimator, z, shift_type
        )
        logits_unshifted = log_density_ratio_under_shift(
            treatment, delta, density_estimator, z, shift_type
        )

        # classification targets are 1 for shifted and 0 for unshifted
        logits = torch.cat([logits_shifted, logits_unshifted])
        tgts = torch.cat([torch.ones_like(treatment), torch.zeros_like(treatment)])

        # make loss and return
        if self.multiscale:
            sig = self.lsig[ix].clamp(max=10).exp()
        else:
            sig = self.lsig.clamp(max=10).exp()
        const = 1.0 / (1e-10 + sig**2)
        loss = const * F.binary_cross_entropy_with_logits(logits, tgts)
        losses.append(loss)

        return torch.stack(losses).sum()

    def prior(self):
        sig = self.lsig.clamp(max=10).exp()
        return -dists.HalfCauchy(1.0).log_prob(sig).mean()


class VarianceRegularizer(nn.Module):
    def __init__(self, delta_list, multiscale: bool = True, fit_scale: bool = True) -> None:
        super().__init__()
        self.register_buffer("multiscale", torch.tensor(multiscale))
        if multiscale:
            self.lsig = nn.Parameter(torch.zeros_like(delta_list), requires_grad=fit_scale)
        else:
            self.lsig = nn.Parameter(torch.tensor(0.0), requires_grad=fit_scale)
        self.register_buffer("delta_list", delta_list)

    def loss(self, treatment, density_estimator, z, shift_type):
        losses = []

        # sample delta at random for each element
        ix = torch.randint(
            high=len(self.delta_list),
            size=(treatment.shape[0], ),
            device=treatment.device
        )
        delta = self.delta_list[ix]

        logits = log_density_ratio_under_shift(
            treatment, delta, density_estimator, z, shift_type
        )

        # minimizing the differnces between contiguous observations
        # is equivalent to variance minimization using u-statistics
        approx_variance = 0.5 * logits.diff().pow(2).mean()

        # make loss and return
        if self.multiscale:
            sig = self.lsig[ix].clamp(max=10).exp()
        else:
            sig = self.lsig.clamp(max=10).exp()
        const = 1.0 / (1e-10 + sig**2)
        loss = const * approx_variance
        losses.append(loss)

        return torch.stack(losses).sum()

    def prior(self):
        sig = self.lsig.clamp(max=10).exp()
        return -dists.HalfCauchy(1.0).log_prob(sig).sum()
