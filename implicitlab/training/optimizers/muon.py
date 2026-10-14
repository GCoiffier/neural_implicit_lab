import torch
import torch.distributed as dist


def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    [https://kellerjordan.github.io/posts/muon/](https://kellerjordan.github.io/posts/muon/)

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. For efficient orthogonalization we use a Newton-Schulz iteration, which has the
    advantage that it can be stably run in bfloat16 on the GPU.

    Muon should only be used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases should be optimized using a standard method such as AdamW.
    Hidden convolutional weights can be trained using Muon by viewing them as 2D and then
    collapsing their last 3 dimensions.

    Muon has been shown to improve the accuracy of neural implicit representations.

    References:
        _Optimizing Rank for High-Fidelity Implicit Neural Representations, J. McGinnis et al., 2026

    Args:
        lr: The learning rate, in units of spectral norm per update.
        weight_decay: The AdamW-style weight decay.
        momentum: The momentum. A value of 0.95 here is usually fine.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
            for base_i in range(len(params))[::dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])

        return loss


class SingleDeviceMuon(torch.optim.Optimizer):
    """
    Muon variant for usage in non-distributed settings.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """
    Distributed Muon variant that can be used for all parameters in the network, since it runs an
    internal AdamW for the parameters that are not compatible with Muon. The user must manually
    specify which parameters shall be optimized with Muon and which with Adam by passing in a
    list of param_groups with the `use_muon` flag set.

    The point of this class is to allow the user to have a single optimizer in their code, rather
    than having both a Muon and an Adam which each need to be stepped.

    You can see an example usage below:

    [https://github.com/KellerJordan/modded-nanogpt/blob/master/records/052525_MuonWithAuxAdamExample/b01550f9-03d8-4a9c-86fe-4ab434f1c5e0.txt#L470](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/052525_MuonWithAuxAdamExample/b01550f9-03d8-4a9c-86fe-4ab434f1c5e0.txt#L470)
    
    Example:
        ```python
        hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
        embed_params = [p for n, p in model.named_parameters() if "embed" in n]
        scalar_params = [p for p in model.parameters() if p.ndim < 2]
        head_params = [model.lm_head.weight]

        from muon import MuonWithAuxAdam
        adam_groups = [dict(params=head_params, lr=0.22), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
        adam_groups = [dict(**g, betas=(0.8, 0.95), eps=1e-10, use_muon=False) for g in adam_groups]
        muon_group = dict(params=hidden_matrix_params, lr=0.05, momentum=0.95, use_muon=True)
        param_groups = [*adam_groups, muon_group]
        optimizer = MuonWithAuxAdam(param_groups)
        ```
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        if p.grad is None:
                            # continue
                            p.grad = torch.zeros_like(p)  # Force synchronization
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam.

    Parameters are updated with `torch._foreach_*` operations, and the Newton-Schulz orthogonalization is batched over all parameters sharing the same shape. This gives the same updates as applying `muon_update` and `adam_update` to each parameter separately, with far fewer kernel launches.
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)  # Force synchronization
            if group["use_muon"]:
                self._muon_step(group)
            else:
                self._adam_step(group)

        return loss

    def _muon_step(self, group):
        params_by_shape = dict()
        for p in group["params"]:
            state = self.state[p]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(p)
            params_by_shape.setdefault(p.shape, []).append(p)
        beta = group["momentum"]
        for shape, params in params_by_shape.items():
            grads = [p.grad for p in params]
            momentums = [self.state[p]["momentum_buffer"] for p in params]
            # Same as muon_update with nesterov=True, for all parameters of this shape at once
            torch._foreach_lerp_(momentums, grads, 1 - beta)
            torch._foreach_lerp_(grads, momentums, beta)
            updates = torch.stack([g.view(len(g), -1) if g.ndim == 4 else g for g in grads])
            updates = zeropower_via_newtonschulz5(updates, steps=5)
            updates *= max(1, updates.size(-2) / updates.size(-1))**0.5
            updates = [u.reshape(shape).to(p.dtype) for u, p in zip(updates.unbind(0), params)]
            torch._foreach_mul_(params, 1 - group["lr"] * group["weight_decay"])
            torch._foreach_add_(params, updates, alpha=-group["lr"])

    def _adam_step(self, group):
        params = group["params"]
        if len(params) == 0: return
        for p in params:
            state = self.state[p]
            if len(state) == 0:
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] = 0
            state["step"] += 1
        beta1, beta2 = group["betas"]
        grads = [p.grad for p in params]
        exp_avgs = [self.state[p]["exp_avg"] for p in params]
        exp_avg_sqs = [self.state[p]["exp_avg_sq"] for p in params]
        # Same as adam_update, for all parameters of the group at once
        torch._foreach_lerp_(exp_avgs, grads, 1 - beta1)
        torch._foreach_lerp_(exp_avg_sqs, torch._foreach_mul(grads, grads), 1 - beta2)
        bias_corrections1 = [1 - beta1**self.state[p]["step"] for p in params]
        bias_corrections2 = [1 - beta2**self.state[p]["step"] for p in params]
        denom = torch._foreach_sqrt(torch._foreach_div(exp_avg_sqs, bias_corrections2))
        torch._foreach_add_(denom, group["eps"])
        updates = torch._foreach_div(torch._foreach_div(exp_avgs, bias_corrections1), denom)
        torch._foreach_mul_(params, 1 - group["lr"] * group["weight_decay"])
        torch._foreach_add_(params, updates, alpha=-group["lr"])
