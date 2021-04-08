import torch
import torch.nn.functional as F
import numpy as np

def accuracy(model, batches):
    with torch.no_grad():
        n, acc = 0, 0
        for batch in batches:
            X, y = batch['input'], batch['target']
            output = model(X)
            acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return acc / n

# Certifiably robust accuracies from Lipschitz-Margin Training
# https://arxiv.org/abs/1802.04034
def cert_stats(model, batches, cert, full=False):
    cert_right = 0.
    cert_wrong = 0.
    insc_right = 0.
    insc_wrong = 0.
    n = 0

    for batch in batches:
        X, y = batch['input'], batch['target']
        yhat = model(X)
        correct = yhat.max(1)[1] == y
        margins = torch.sort(yhat, 1)[0]
        certified = (margins[:,-1] - margins[:,-2]) > cert
        n += len(batch['target'])
        cert_right += torch.sum(correct & certified).item()
        cert_wrong += torch.sum(~correct & certified).item()
        insc_right += torch.sum(correct & ~certified).item()
        insc_wrong += torch.sum(~correct & ~certified).item()
        if not full:
            break 

    cert_right /= n
    cert_wrong /= n
    insc_right /= n
    insc_wrong /= n

    return cert_right, cert_wrong, insc_right, insc_wrong

############################################################
# From: https://github.com/ColinQiyangLi/LConvNet
############################################################

def get_margin_factor(p):
    if p == "inf":
        return 2.0
    return 2.0 ** ((p - 1) / p)

def margin_loss(y_pred, y, eps, p, l_constant, order=1):
    margin = eps * get_margin_factor(p) * l_constant
    return F.multi_margin_loss(y_pred, y, margin=margin, p=order)

####################################################################
# From: https://github.com/tml-epfl/understanding-fast-adv-training
####################################################################

def clamp(X, l=0.0, u=1.0):
    u = torch.cuda.FloatTensor(1).fill_(u)
    l = torch.cuda.FloatTensor(1).fill_(l)
    return torch.max(torch.min(X, u), l)

def attack_pgd(model, X, y, eps, alpha, opt, half_prec, attack_iters, n_restarts, rs=True, verbose=False,
               linf_proj=True, l2_proj=False, l2_grad_update=False, cuda=True, lossf=F.cross_entropy):
    if n_restarts > 1 and not rs:
        raise ValueError('no random step and n_restarts > 1!')
    max_loss = torch.zeros(y.shape[0])
    max_delta = torch.zeros_like(X)
    if cuda:
        max_loss, max_delta = max_loss.cuda(), max_delta.cuda()
    for i_restart in range(n_restarts):
        delta = torch.zeros_like(X)
        if cuda:
            delta = delta.cuda()
        if attack_iters == 0:
            return delta.detach()
        if rs:
            delta.uniform_(-eps, eps)

        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(clamp(X + delta, 0, 1))
            loss = lossf(output, y)
            if half_prec:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
                    delta.grad.mul_(loss.item() / scaled_loss.item())
            else:
                loss.backward()
            grad = delta.grad.detach()

            if not l2_grad_update:
                delta.data = delta + alpha * torch.sign(grad)
            else:
                delta.data = delta + alpha * grad / (grad**2).sum([1, 2, 3], keepdim=True)**0.5

            delta.data = clamp(X + delta.data, 0, 1) - X

            assert linf_proj ^ l2_proj, "cannot be both linf and l2"

            if linf_proj:
                delta.data = clamp(delta.data, -eps, eps)
            if l2_proj:
                delta_norms = (delta.data**2).sum([1, 2, 3], keepdim=True)**0.5
                delta.data = eps * delta.data / torch.max(eps*torch.ones_like(delta_norms), delta_norms)
            delta.grad.zero_()

        with torch.no_grad():
            output = model(X + delta)
            all_loss = lossf(output, y, reduction='none')  # .detach()  # prevents a memory leak
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]

            max_loss = torch.max(max_loss, all_loss)
            if verbose:  # and n_restarts > 1:
                print('Restart #{}: best loss {:.3f}'.format(i_restart, max_loss.mean()))
    max_delta = clamp(X + max_delta) - X
    return max_delta


def rob_acc(batches, model, eps, pgd_alpha, opt, half_prec, attack_iters, n_restarts, rs=True, linf_proj=True,
            l2_grad_update=False, corner=False, print_fosc=False, verbose=False, cuda=True, lossf=F.cross_entropy):
    n_corr_classified, train_loss_sum, n_ex = 0, 0.0, 0
    pgd_delta_list, pgd_delta_proj_list = [], []
    for i, batch in enumerate(batches):
        X, y = batch['input'], batch['target']
        pgd_delta = attack_pgd(model, X, y, eps, pgd_alpha, opt, half_prec, attack_iters, n_restarts, rs=rs, verbose=verbose, linf_proj=linf_proj, l2_grad_update=l2_grad_update, l2_proj=l2_grad_update, cuda=cuda, lossf=lossf)
        if corner:
            pgd_delta = clamp(X + eps * torch.sign(pgd_delta)) - X
        pgd_delta_proj = clamp(X + eps * torch.sign(pgd_delta)) - X  # needed just for investigation

        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = lossf(output, y)
        n_corr_classified += (output.max(1)[1] == y).sum().item()
        train_loss_sum += loss.item() * y.size(0)
        n_ex += y.size(0)
        pgd_delta_list.append(pgd_delta.cpu().numpy())
        pgd_delta_proj_list.append(pgd_delta_proj.cpu().numpy())

    robust_acc = n_corr_classified / n_ex
    avg_loss = train_loss_sum / n_ex
    pgd_delta_np = np.vstack(pgd_delta_list)
    return robust_acc, avg_loss, pgd_delta_np

####################################################################
# Based on: https://github.com/yangarbiter/robust-local-lipschitz
####################################################################

def empirical_local_lipschitzity(model, batches, eps=36.0/255.0, early_stop=False, ret_delta=False):
    norms = lambda X, p: X.view(X.shape[0], -1).norm(p=p, dim=1)
    alpha = eps/4.0

    total_loss = 0.0
    total_batches = 0
    for batch in batches:
        X, y = batch['input'], batch['target']
        
        delta = torch.zeros_like(X)
        delta.uniform_(-eps, eps)   
        delta.data = torch.min(torch.max(delta.detach(), -X), 1 - X)
        delta.data *= eps / norms(delta.detach(), p=2)[:, None, None, None].clamp(min=eps)
        for _ in range(10):

            delta.requires_grad = True

            loss = (norms(model(X + delta) - model(X), p=2) / norms(delta+1e-6, p=2)).mean()
            loss.backward()

            grad = delta.grad.detach()

            delta.data += alpha * grad / norms(grad, p=2)[:, None, None, None]
            delta.data = torch.min(torch.max(delta.detach(), -X), 1 - X)
            delta.data *= eps / norms(delta.detach(), p=2)[:, None, None, None].clamp(min=eps)

            delta.grad.zero_()
        
        lossmax = (norms(model(X + delta) - model(X), p=2) / norms(delta+1e-6, p=2)).max()
        total_loss += lossmax.detach()
        total_batches += 1
        if early_stop:
            break

    if not ret_delta:
        return total_loss / total_batches
    else:
        return total_loss / total_batches, (X, delta)

