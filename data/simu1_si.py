import torch
from torch import cat, stack

# Mauricio Tec changes:
# renaming x_t -> make_treatment
#          t_x_y -> make_outcome
# adding the noise as an input,
# removing x_t_link which is simply 
# the sigmoid function, so merging with treatment

# but vectorized for speed.

def pmax(*args):
    return stack(list(args), dim=1).amax(dim=1)

def make_treatment(x, noise):
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    x4 = x[:, 3]
    x5 = x[:, 4]
    logits = (10. * pmax(x1, x2, x3).sin() + pmax(x3, x4, x5)**3) / (1. + (x1 + x5)**2) + \
        (0.5 * x3).sin() * (1. + (x4 - 0.5 * x3).exp()) + \
        x3**2 + 2. * x4.sin() + 2*x5 - 6.5
    t = (logits + noise).sigmoid()
    return t

def make_outcome(t, x, noise):
    # only x1, x3, x4 are useful
    x1 = x[:, 0]
    x3 = x[:, 2]
    x4 = x[:, 3]
    x6 = x[:, 5]
    y = ((t-0.5) * 2*torch.pi).cos() * (t**2 + (4*pmax(x1, x6)**3)/(1. + 2*x3**2)*x4.sin())
    return y + noise

def simu_data1(n_train, n_test, d_list):
    """
    delta_std is the number of standard deviations to reduce from the treatment
    """
    # -- should be same as vcnet code, but vectorized -- #
    train_ix = torch.arange(0, n_train)
    test_ix = torch.arange(n_train, n_train + n_test)
    n = n_train + n_test

    noise_t = 0.5 * torch.randn(n)
    noise_y = 0.5 * torch.randn(n)
   
    x = torch.rand((n, 6))
    t = make_treatment(x, noise_t)
    y = make_outcome(t, x, noise_y)

    train_matrix = cat([t[train_ix, None], x[train_ix], y[train_ix, None]], dim=1)
    test_matrix = cat([t[test_ix, None], x[test_ix], y[test_ix, None]], dim=1)

    # -- specific to stochastic interventions -- #
    # estimate the insample srf
    # logit and sigmoid are inverse operations
    logits = t.logit()  # logits only when treatment in (0, 1)
    delta_scale = logits.std()

    shifted_treatments = [(logits - delta_scale * d).sigmoid() for d in d_list]
    y_cf = stack([make_outcome(t, x, noise_y) for t in shifted_treatments], dim=1)
    srf_train = y_cf[train_ix, :].mean(0)
    srf_test = y_cf[test_ix, :].mean(0)

    return {
        "train_matrix": train_matrix,
        "test_matrix": test_matrix,
        "srf_train": srf_train,
        "srf_test": srf_test,
        "delta_scale": delta_scale
    }




