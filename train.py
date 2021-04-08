import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
import time
import os
import utils
# how to change the batch size:
# os.environ['batch_size'] = '256'
import data
from layers import *
from models import *


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='KWLarge', choices=['KWLarge', 'ResNet9', 'WideResNet'])
parser.add_argument('--conv', default='CayleyConv', 
                    choices=['CayleyConv', 'BCOP', 'RKO', 'SVCM', 'OSSN', 'PlainConv'])
parser.add_argument('--linear', default='CayleyLinear', 
                    choices=['CayleyLinear', 'BjorckLinear', 'nn.Linear'])
parser.add_argument('--lr_max', default=0.01, type=float)
parser.add_argument('--stddev', action='store_true')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--eps', default=36.0, type=float)

args = parser.parse_args()

eps = args.eps / 255.0
alpha = eps / 4.0

_model = eval(args.model)
conv = eval(args.conv)
linear = eval(args.linear)

model = nn.Sequential(
    Normalize(data.mu, data.std if args.stddev else 1.0),
    _model(conv=conv, linear=linear)
).cuda()

model_name = args.model
epochs = args.epochs
lr_max = args.lr_max

# for SVCM projections
proj_nits = 100

# lr schedule: superconvergence
lr_schedule = lambda t: np.interp([t], [0, epochs*2//5, epochs*4//5, epochs], [0, lr_max, lr_max/20.0, 0])[0]

# optimizer: Adam
opt = optim.Adam(model.parameters(), lr=lr_max, weight_decay=0)

# loss: multi-margin loss
criterion = lambda yhat, y: utils.margin_loss(yhat, y, 0.5, 1.0, 1.0)

for epoch in range(epochs):
    start = time.time()
    train_loss, acc, n = 0, 0, 0
    for i, batch in enumerate(data.train_batches):
        X, y = batch['input'], batch['target']
        
        lr = lr_schedule(epoch + (i + 1)/len(data.train_batches))
        opt.param_groups[0].update(lr=lr)
        
        output = model(X)
        loss = criterion(output, y)

        opt.zero_grad()
        loss.backward()
        opt.step()
        
        train_loss += loss.item() * y.size(0)
        acc += (output.max(1)[1] == y).sum().item()
        n += y.size(0)
        
        # for SVCM projections
        if i % proj_nits == 0 or i == len(data.train_batches) - 1:
            for m in model.modules():
                if hasattr(m, '_project'):
                    m._project()
    
    if (epoch + 1) % 10 == 0:
        l_emp = utils.empirical_local_lipschitzity(model, data.test_batches, early_stop=True).item()
        print('[{}] --- Empirical Lipschitzity: {}'.format(args.model, l_emp))

    print(f'[{args.model}] Epoch: {epoch} | Train Acc: {acc/n:.4f}, Test Acc: {utils.accuracy(model, data.test_batches):.4f}, Time: {time.time() - start:.1f}, lr: {lr:.6f}')

if not args.stddev:
    print('[{}] (PROVABLE) Certifiably Robust (eps: {:.4f}): {:.4f}, Cert. Wrong: {:.4f}, Insc. Right: {:.4f}, Insc. Wrong: {:.4f}'.format(args.model, eps, *utils.cert_stats(model, data.test_batches, eps * 2**0.5, full=True)))

print('[{}] (EMPIRICAL) Robust accuracy (eps: {:.4f}): {}'.format(args.model, eps, utils.rob_acc(data.test_batches, model, eps, alpha, opt, False, 10, 1, linf_proj=False, l2_grad_update=True)[0]))
            