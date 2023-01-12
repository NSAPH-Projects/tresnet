import os

import pandas as pd
import numpy as np

import torch


# TODO:
# 1. Check the indices of outcome, treatment and confounders [DONE]
# Check the shape of the outcome of the baseline model
# 2. Implement loss function
# 3. Implement optimizer and put all the criteria in
# 4. Document all the datasets
# 5. Run experiments

from models.VCNet import VCNet
from dataset.dataset import get_iter

# TODO: Change shuffling in dataloader (get_iter) to True here and in legacy

seed = 1234
torch.manual_seed(seed)

load_path = "dataset/simu1/eval/0"
num_epoch = 800
lr_type = "fixed"
wd = 5e-3  # weight decay
momentum = 0.9
# targeted regularization optimizer
tr_wd = 5e-3  # weight decay


data = pd.read_csv(load_path + "/train.txt", header=None, sep=" ")
train_matrix = torch.from_numpy(data.to_numpy()).float()
data = pd.read_csv(load_path + "/test.txt", header=None, sep=" ")
test_matrix = torch.from_numpy(data.to_numpy()).float()
data = pd.read_csv(load_path + "/t_grid.txt", header=None, sep=" ")
t_grid = torch.from_numpy(data.to_numpy()).float()

train_loader = get_iter(train_matrix, batch_size=500, shuffle=False)

density_estimator_config = [(6, 50, 1), (50, 50, 1)]
num_grid = 10

pred_head_config = [
    (
        50,
        50,
        1,
    ),
    (50, 1, 1),
]
spline_degree = 2
spline_knots = [0.33, 0.66]


# TODO: define model

# torch.manual_seed(seed)
model = VCNet(
    density_estimator_config, num_grid, pred_head_config, spline_degree, spline_knots
)
model._initialize_weights()
# TODO: restore the weight initialization

# print(model)
# print(list(model.parameters()))
# raise


is_target_reg = True

for idx, item in enumerate(train_loader):
    t = item["treatment"]
    x = item["covariates"]
    y = item["outcome"]

    if is_target_reg:
        # optimizer.zero_grad()
        out = model.forward(t, x)
        print("Output", out)
        print("Output shape", out["predicted_outcome"].shape)
        print("The full model works")
        raise
        # TODO: remove the raise from here
        trg = TargetReg(t)
        loss = criterion(out, y, alpha=alpha) + criterion_TR(out, trg, y, beta=beta)
        loss.backward()
        optimizer.step()

        tr_optimizer.zero_grad()
        out = model.forward(t, x)
        trg = TargetReg(t)
        tr_loss = criterion_TR(out, trg, y, beta=beta)
        tr_loss.backward()
        tr_optimizer.step()
