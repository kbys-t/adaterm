# coding:utf-8

import math
import torch
from torch.optim import Optimizer

class AdaTerm(Optimizer):
    r"""Implements an original optimizer
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.9),
                 eps=1e-5, weight_decay=0.0, amsgrad=False,
                 ini_dof=0.0, uncentered=False, adabias_corr=False,
                 k_dof=1.0, beta_dof=0.9):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= amsgrad <= 1.0:
            raise ValueError("Invalid amsgrad parameter: {}".format(amsgrad))
        if not 0.0 <= ini_dof:
            raise ValueError("Invalid ini_dof parameter: {}".format(ini_dof))
        if not isinstance(uncentered, bool):
            raise ValueError("Invalid uncentered parameter: {}".format(uncentered))
        if not isinstance(adabias_corr, bool):
            raise ValueError("Invalid adabias_corr parameter: {}".format(adabias_corr))
        if not ((k_dof > 0.0) or (k_dof == math.inf)):
            raise ValueError("Invalid degrees of freedom scale factor: {}".format(k_dof))
        if not 0.0 <= beta_dof <= 1.0:
            raise ValueError("Invalid beta parameter for dof optimisation: {}".format(beta_dof))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        ini_dof=ini_dof, uncentered=uncentered, adabias_corr=adabias_corr,
                        k_dof=k_dof, beta_dof=beta_dof, inf_dof=(k_dof == math.inf), optim_dof=(beta_dof < 1.0))
        super(AdaTerm, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdaTerm, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdaTerm, just as Adam, does not support sparse gradients, please consider SparseAdam instead")
                amsgrad = group["amsgrad"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                k_dof = group["k_dof"]
                inf_dof = group["inf_dof"]
                optim_dof = group["optim_dof"]
                adabias_corr = group["adabias_corr"]

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_var"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_eps"] = torch.zeros_like(p, memory_format=torch.preserve_format).add_(eps**2)
                    if amsgrad:
                        state["max_exp_var"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if not inf_dof:
                        state["dof"] = max(k_dof + eps, group["ini_dof"])
                    state["adabias_correction1"] = 0.0
                    state["adabias_correction2"] = 0.0

                exp_avg, exp_var, exp_eps = state["exp_avg"], state["exp_var"], state["exp_eps"]
                adabias_correction1, adabias_correction2 = state["adabias_correction1"], state["adabias_correction2"]
                if amsgrad:
                    max_exp_var = state["max_exp_var"].mul_(amsgrad)

                if group["weight_decay"]:
                    grad.add_(p, alpha=group["weight_decay"])

                # compute coeffs
                diff = grad.sub(exp_avg).square_()
                if inf_dof:
                    tau1, tau2 = 1.0 - beta1, 1.0 - beta2
                    delta = eps**2
                else:
                    dof = state["dof"]
                    dd = dof + 1.0
                    sigma_sq = exp_var.add(exp_eps)
                    D_ = diff.div(sigma_sq).mean().item()
                    delta = diff.sub(sigma_sq, alpha=D_).div_(dof).clamp_(min=eps**2)
                    w_ = dd / (dof + D_)
                    w_max = dd / dof
                    tau = w_ / w_max
                    tau1 = tau * (1.0 - beta1)
                    tau2 = tau * (1.0 - beta2)
                    if optim_dof:
                        w_dof = w_ - math.log(max(w_, torch.finfo(torch.float32).tiny))
                        w_max = max(w_max - math.log(w_max), - math.log(torch.finfo(torch.float32).tiny))
                        tau_dof = w_dof / w_max * (1.0 - group["beta_dof"])
                        dof_tar = ((dof + 2.0) / dd + dof) / w_dof * (1.0 - k_dof / dof) + (k_dof + eps)

                # update statistics
                exp_avg.mul_(1.0 - tau1).add_(grad, alpha=tau1)
                exp_var.mul_(1.0 - tau2).add_(diff, alpha=tau2)
                exp_eps.mul_(1.0 - tau2).add_(delta, alpha=tau2)
                if (not inf_dof) and optim_dof:
                    state["dof"] = (1.0 - tau_dof) * dof + tau_dof * dof_tar

                # update step and bias corrections
                state["step"] += 1
                if adabias_corr:
                    state["adabias_correction1"] = adabias_correction1 * (1.0 - tau1) + tau1
                    state["adabias_correction2"] = adabias_correction2 * (1.0 - tau2) + tau2
                    bias_correction1, bias_correction2 = state["adabias_correction1"], state["adabias_correction2"]
                else:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]

                # set denominator
                deno = exp_var.addcmul(exp_avg, exp_avg, value=group["uncentered"])
                if amsgrad:
                    deno = torch.maximum(max_exp_var, deno, out=max_exp_var)

                # update parameter
                nume = exp_avg.div(bias_correction1)
                deno = deno.add(exp_eps).div_(bias_correction2).sqrt_()
                p.addcdiv_(nume, deno, value=-group["lr"])

        return loss
