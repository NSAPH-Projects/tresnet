import os

import pandas as pd
import numpy as np

import torch

from utils.loss import criterion


# TODO:
# 1. Check the indices of outcome, treatment and confounders [DONE]
# Check the shape of the outcome of the baseline model [Done]
# 2. Implement loss function
# 3. Implement optimizer and put all the criteria in
# 4. Document all the datasets
# 5. Run experiments

from models.models import VCNet
from models.modules import TargetedRegularizerCoeff
from dataset.datasets import get_iter

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
pred_spline_degree = 2
pred_spline_knots = [0.33, 0.66]

loss_tr_knots = list(np.arange(0.1, 1, 0.1))
loss_tr_degree = 2

targeted_regularizer = TargetedRegularizerCoeff(loss_tr_degree, loss_tr_knots)
targeted_regularizer._initialize_weights()


# define model

torch.manual_seed(seed)
model = VCNet(
    density_estimator_config,
    num_grid,
    pred_head_config,
    pred_spline_degree,
    pred_spline_knots,
)
model._initialize_weights()


# Optimizer
lr_type = "fixed"
wd = 5e-3
momentum = 0.9
# targeted regularization optimizer
tr_wd = 5e-3

init_lr = 0.0001
alpha = 0.5
tr_init_lr = 0.001
beta = 1.0


optimizer = torch.optim.SGD(
    model.parameters(),
    lr=init_lr,
    momentum=momentum,
    weight_decay=wd,
    nesterov=True,
)

tr_optimizer = torch.optim.SGD(
    targeted_regularizer.parameters(), lr=tr_init_lr, weight_decay=tr_wd
)
# print(model)
# print(list(model.parameters()))
# raise

is_target_reg = True

num_epoch = 800

for epoch in range(num_epoch):
    for idx, item in enumerate(train_loader):
        t = item["treatment"]
        x = item["covariates"]
        y = item["outcome"]

        if is_target_reg:
            optimizer.zero_grad()
            model_output = model.forward(t, x)
            targeted_regularizer_coeff = targeted_regularizer(t)
            loss = criterion(
                model_output, y, target_reg_coeff=targeted_regularizer_coeff
            )
            loss["total_loss"].backward()
            optimizer.step()

            tr_optimizer.zero_grad()
            model_output = model.forward(t, x)
            targeted_regularizer_coeff = targeted_regularizer(t)
            loss = criterion(
                model_output, y, target_reg_coeff=targeted_regularizer_coeff
            )
            loss["targeted_reg_loss"].backward()
            tr_optimizer.step()
        if epoch % 100 == 0:
            print("current epoch: ", epoch)
            print("loss value: ", round(loss["total_loss"].data.item(), 4))
