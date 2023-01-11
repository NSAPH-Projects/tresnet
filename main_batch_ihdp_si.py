import torch
import math
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn

import os
import json
from models.DisCri import DisCri

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from models.dynamic_net import Vcnet, Drnet, TR, Dynamic_FC
from models.itermidate_models import VcnetAtt, VcnetAttv2
from data.data import get_iter
from utils.eval import si_curve
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, MultiStepLR


import argparse
os.environ["CUDA_VISIBLE_DEVICES"]="3"
def adjust_learning_rate(optimizer, init_lr, epoch, lr_type='cos', num_epoch=50):
    if lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * epoch / num_epoch))
    elif lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = init_lr * (decay ** (epoch // step))
    elif lr_type == 'fixed':
        lr = init_lr
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def save_checkpoint(state, model_name='', checkpoint_dir='.'):
    filename = os.path.join(checkpoint_dir, model_name + '_ckpt.pth.tar')
    print('=> Saving checkpoint to {}'.format(filename))
    torch.save(state, filename)

# criterion
def criterion(out, y, alpha=0.5, epsilon=1e-6):
    return F.mse_loss(out[1].squeeze(),  y.squeeze()) # - alpha * torch.log(out[0] + epsilon).mean()

def criterion_SI(out, gd, mud, trg, y, beta=1., epsilon=1e-6):
    # out[1] is Q
    # out[0] is g
    return beta * ((y.squeeze() - trg.squeeze() * gd / (out[0] + epsilon) - mud.squeeze()) ** 2).mean()

def criterion_TR(out, trg, y, beta=1., epsilon=1e-6):
    # out[1] is Q
    # out[0] is g
    return beta * ((y.squeeze() - trg.squeeze() / (out[0].squeeze() + epsilon) - out[1].squeeze()) ** 2).mean()


def neg_guassian_likelihood(d, u):
    """return: -N(u; mu, var)"""
    B, dim = u.shape[0], 1
    assert (d.shape[1] == dim * 2)
    mu, logvar = d[:, :dim], d[:, dim:]
    return 0.5 * (((u - mu) ** 2) / torch.exp(logvar) + logvar).mean()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train with simulate data')

    # i/o
    parser.add_argument('--data_dir', type=str, default='dataset/ihdp/ihdp_si/tr_h_1.75_te_l_0.25_h2.0', help='dir of eval dataset')
    parser.add_argument('--save_dir', type=str, default='logs/ihdp_si/tr_h_1.75_te_l_0.25_h2.0/', help='dir to save result')

    # common
    parser.add_argument('--num_dataset', type=int, default=1, help='num of datasets to train')
    parser.add_argument('--h', type=int, default=1.75, help='the maximal value of treatmemts')
    parser.add_argument('--clip', type=int, default=3., help='gradient clip')

    # training
    parser.add_argument('--n_epochs', type=int, default=1500, help='num of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='num of epochs to train')
    parser.add_argument('--d_dim', type=int, default=1, help='num of epochs to train')
    parser.add_argument('--alpha', type=float, default=0.5, help='tradeoff parameter for advesarial loss')
    parser.add_argument('--p', type=int, default=1, help='dim for outputs of treatments discriminator, 1 for value, 2 for mean and var')

    # print train info
    parser.add_argument('--verbose', type=int, default=200, help='print train info freq')
    parser.add_argument('--plt_adrf', type=bool, default=True, help='whether to plot adrf curves.')

    # targeted reg type
    parser.add_argument('--treg', type=str, default='si', choices=('ad', 'tr', 'si'))

    args = parser.parse_args()
    dev = "cpu"

    # fixed parameter for optimizer
    lr_type = 'fixed'
    wd = 5e-3
    momentum = 0.9

    # targeted regularization optimizer
    tr_wd = 5e-3

    num_epoch = args.n_epochs

    # check val loss
    verbose = args.verbose

    # data
    load_path = args.data_dir
    num_dataset = args.num_dataset

    # save
    save_path = args.save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.plt_adrf:
        grid = {} ;MSE = {}; num_dataset=1
    Result = {}

    # for model_name in ['Tarnet', 'Tarnet_tr', 'Drnet', 'Drnet_tr', 'Vcnet', 'Vcnet_tr', 'TransTEE_tr', 'TransTEE']:
    #     Result[model_name]=[]

    method_list = ['Tarnet', 'Drnet', 'Vcnet', 'TransTEE']
    if args.treg is not None:
        method_list = ["_".join([m, args.treg]) for m in method_list]

    for model_name in method_list:
        Result[model_name]=[]
        # import model
        if model_name.startswith('Vcnet'):
            cfg_density = [(25, 50, 1, 'relu'), (50, 50, 1, 'relu')]
            num_grid = 10
            cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
            degree = 2
            knots = [0.33, 0.66]
            model = Vcnet(cfg_density, num_grid, cfg, degree, knots)
            model._initialize_weights()

        elif model_name.startswith('VcnetAtt'):
            cfg_density = [(25, 50, 1, 'relu'), (50, 50, 1, 'relu')]
            num_grid = 10
            cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
            degree = 2
            knots = [0.33, 0.66]
            model = VcnetAtt(cfg_density, num_grid, cfg, degree, knots)
            model._initialize_weights()
        
        elif model_name.startswith('TransTEE'):
            num_cov=25
            num_t=1
            num_heads = 2
            att_layers = 1
            dropout = 0.0
            init_range_f = 0.1
            init_range_t = 0.2
            embed_size = 10
            model = VcnetAttv2(embed_size=embed_size, num_t=num_t, num_cov=num_cov, num_heads=num_heads, att_layers=att_layers, dropout=dropout, init_range_f=init_range_f, init_range_t=init_range_t)
            model._initialize_weights()

        elif model_name.startswith('Drnet'):
            cfg_density = [(25, 50, 1, 'relu'), (50, 50, 1, 'relu')]
            num_grid = 10
            cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
            isenhance = 1
            model = Drnet(cfg_density, num_grid, cfg, isenhance=isenhance, h=args.h)
            model._initialize_weights()

        elif model_name.startswith('Tarnet'):
            cfg_density = [(25, 50, 1, 'relu'), (50, 50, 1, 'relu')]
            num_grid = 10
            cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
            isenhance = 0
            model = Drnet(cfg_density, num_grid, cfg, isenhance=isenhance) # , h=args.h)
            model._initialize_weights()

        model.to(dev)
        # use Target Regularization
        if model_name.endswith('_lr'):
            isTargetReg = 1
        else:
            isTargetReg = 0

        # best cfg for each model
        beta=1.0
        if  model_name.startswith('Tarnet'):
            init_lr = 0.05
            tr_init_lr = 0.001
            alpha = 1.0

            Result['Tarnet'] = []

        elif  model_name.startswith('Drnet'):
            init_lr = 0.05
            tr_init_lr = 0.001
            alpha = 1.

            Result['Drnet'] = []

        elif  model_name.startswith('Vcnet'):
            init_lr = 0.0005
            tr_init_lr = 0.001
            alpha = 0.5

            Result['Vcnet'] = []
        elif  model_name.startswith('VcnetAtt'):
            init_lr = 0.0005
            tr_init_lr = 0.001
            alpha = 0.5

            Result['VcnetAtt'] = []  
        
        elif  model_name.startswith('TransTEE'):
            init_lr = 0.0005
            tr_init_lr = 0.001
            #init_lr = 0.01
            alpha = 0.5
        
            Result['TransTEE'] = [] 
        # elif  model_name.startswith('TransTEE_tr'):
        #     init_lr = 0.0005
        #     #init_lr = 0.01
        #     alpha = 0.5
        #     tr_init_lr = 0.001
        
        #     Result['TransTEE_tr'] = [] 
        #     # Result['TransTEE_tr_result'] = [] 
        if args.h != 1:
            data_matrix = torch.load(args.data_dir + '/train_matrix.pt')
            test_data_matrix = torch.load(args.data_dir + '/data_matrix.pt')
        else:
            data_matrix = torch.load(args.data_dir + '/data_matrix.pt')
        
        #!! change: load truth of stochastic interventions
        delta = torch.load(args.data_dir + '/delta_grid.pt')
        mu_delta = torch.load(args.data_dir + '/mu_delta_grid.pt')
        counterfactuals = torch.load(args.data_dir + '/counterfactuals_grid.pt')

        for _ in range(num_dataset):
            train_idx = torch.load(args.data_dir + '/eval/' + str(_) + '/idx_train.pt')
            test_idx = torch.load(args.data_dir + '/eval/' + str(_) + '/idx_test.pt')
            train_matrix = data_matrix[train_idx]
            test_matrix = data_matrix[test_idx] if args.h == 1 else test_data_matrix[test_idx]

            # train_matrix, test_matrix, t_grid = simu_data1(500, 200)
            train_loader = get_iter(train_matrix, batch_size=args.batch_size, shuffle=True)
            test_loader = get_iter(test_matrix, batch_size=test_matrix.shape[0], shuffle=False)

            # reinitialize model
            model._initialize_weights()

            # define optimizer
            optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=wd, nesterov=True)

            if args.treg == "tr":
                if 'Att' in model_name or 'TransTEE' in model_name:
                    in_dim = embed_size
                else:
                    in_dim = 50
                TargetReg = DisCri(in_dim, 50, args.p)
                TargetReg.to(dev)
                tr_optimizer = torch.optim.SGD(TargetReg.parameters(), lr=tr_init_lr, weight_decay=tr_wd)

            elif args.treg == "si":
                tr_knots = list(np.arange(0.05, 1, 0.05))
                tr_degree = 2
                TargetReg = TR(tr_degree, tr_knots)
                TargetReg._initialize_weights()
                tr_optimizer = torch.optim.SGD(TargetReg.parameters(), lr=tr_init_lr, weight_decay=tr_wd)
            
            elif args.treg == "si_ratio":
                tr_knots = list(np.arange(0.05, 1, 0.05))
                # tr_degree = 2
                # TargetReg = TR(tr_degree, tr_knots)
                # Ratio = Dynamic_FC(50, 1, degree, knots, act='silu')
                # tr_optimizer = torch.optim.SGD(TargetReg.parameters(), lr=tr_init_lr, weight_decay=tr_wd)
    
    
            print('model : ', model_name)
            for epoch in range(num_epoch):

                for idx, (inputs, y) in enumerate(train_loader):
                    inputs = Variable(inputs.to(dev).detach())
                    y = Variable(y.to(dev).detach())
                    t = inputs[:, 0]
                    x = inputs[:, 1:]

                    if args.treg == "ad":
                        out = model.forward(t, x)

                        set_requires_grad(TargetReg, True)
                        tr_optimizer.zero_grad()
                        trg = TargetReg(out[0].detach())
                        if args.p == 1:
                            loss_D = F.mse_loss(trg.squeeze(), t)
                        elif args.p == 2:
                            loss_D = neg_guassian_likelihood(trg.squeeze(), t)
                        loss_D.backward()
                        if args.clip:
                            torch.nn.utils.clip_grad_norm_(TargetReg.parameters(), args.clip)
                        tr_optimizer.step()
                    
                        set_requires_grad(TargetReg, False)
                        optimizer.zero_grad()
                        trg = TargetReg(out[0])
                        if args.p == 1:
                            loss_D = - F.mse_loss(trg.squeeze(), t)
                        elif args.p == 2:
                            loss_D = neg_guassian_likelihood(trg.squeeze(), t)
                        loss = criterion(out, y, alpha=alpha) - args.alpha * loss_D
                        loss.backward()
                        if args.clip:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                        optimizer.step()
                        adjust_learning_rate(optimizer, init_lr, epoch, lr_type='cos', num_epoch=num_epoch)
                    elif args.treg == "si":
                        optimizer.zero_grad()
                        d = torch.rand(t.shape[0]) # random distributional shift
                        out = model.forward(t, x)
                        gd = d * model.density_estimator_head(t/d, model.hidden)
                        mud = model.Q(torch.cat([(d * t).unsqueeze(1), model.hidden], 1))
                        trg = TargetReg(t)
                        loss = criterion(out, y, alpha=alpha) + criterion_SI(out, gd, mud, trg, y, beta=beta)
                        loss.backward()
                        optimizer.step()

                        tr_optimizer.zero_grad()
                        out = model.forward(t, x)
                        trg = TargetReg(t)
                        tr_loss = criterion_SI(out, trg, y, beta=beta)
                        tr_loss.backward()
                        tr_optimizer.step()
                        adjust_learning_rate(optimizer, init_lr, epoch, lr_type='cos', num_epoch=num_epoch)
                    else:
                        optimizer.zero_grad()
                        out = model.forward(t, x)
                        loss = criterion(out, y, alpha=alpha)
                        loss.backward()
                        if args.clip:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                        optimizer.step()
                        adjust_learning_rate(optimizer, init_lr, epoch, lr_type='cos', num_epoch=num_epoch)

                if epoch % verbose == 0:
                    print('current epoch: ', epoch)
                    print('loss: ', loss.data)
                    if isTargetReg:
                        print('adv_loss: ', loss_D.data)
                    model.eval()
                    #!! change: use si_curve instead of si
                    mu_delta_hat, mse = si_curve(model, test_matrix, delta, mu_delta)
                    model.train()
                    print('current test loss: ', mse)

            model.eval()
            #out = model.forward(t, x)
            #!! change: use si_curve instead of si
            mu_delta_hat, mse = si_curve(model, test_matrix, delta, mu_delta)

            mse = float(mse)
            print('current loss: ', float(loss.data))
            print('current test loss: ', mse)
            # print('current pred var', torch.std(t_grid_hat).item())

            Result[model_name].append(mse)
            Result['TransTEE_tr_result'] = [np.mean(Result['TransTEE_tr']), np.std(Result['TransTEE_tr'])]

            if args.plt_adrf:
                MSE[model_name, _] = mse
                grid[model_name, _] = mu_delta_hat

            if not args.plt_adrf:
                with open(save_path + '/p' + str(args.p) + str(args.alpha) + 'baselines.json', 'w') as fp:
                    json.dump(Result, fp)

    if args.plt_adrf:
        plt.figure(figsize=(5, 5))
        plt.rcParams["font.family"] = "Times New Roman"
        c1 = 'gold'
        c2 = 'grey'
        c3 = '#d7191c'
        c4 = 'red'
        c0 = '#2b83ba'

        delta_pc = (delta - 1)*100
        plt.plot(delta_pc, mu_delta, marker='', ls='-', label='DSRF', linewidth=4, color=c1)
        plt.ylim(mu_delta.min() - 2, mu_delta.max() + 2)
        plt.xlabel('% decrease from observed treatment')
        plt.ylabel('Average response')
        plt.title("Continuous IHDP dataset")

        colors = ['#abdda4', c2, '#2b83ba', '#d7191c']
        for i, (key, val) in enumerate(grid.items()):
            plt.scatter(delta_pc, val, marker='h', label=key, alpha=0.5, color=colors[i], s=15)
            print('x=', x, '\ny=', y)
        
        plt.grid()
        plt.legend()

        plt.savefig(save_path + '/' +"ihdp_si_curve.pdf", bbox_inches='tight')
        print(MSE)
