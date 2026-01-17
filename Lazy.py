import math
import torch

@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class LazyKimiMuon(torch.optim.Optimizer):
    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        skip_steps=10,
        adamw_params=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
    ):
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            skip_steps=skip_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        
        for p in muon_params:
            assert p.ndim == 2, f"Muon params must be 2D, got {p.ndim}"
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            skip_steps = group["skip_steps"]
            
            # --- Muon 组更新 ---
            muon_params_list = [p for p in group["params"] if self.state[p].get("use_muon", False)]
            
            for p in muon_params_list:
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                
                state = self.state[p]
                
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                    state["step"] = 0
                    state["spectral_gain"] = torch.tensor(1.0, device=p.device)
                    # EF (Error Feedback) 所需状态
                    state["accum_vel"] = torch.zeros_like(g)
                    state["cache_params"] = torch.zeros_like(p)
                    state["EF_momentum_buffer"] = torch.zeros_like(g)

                state["step"] += 1
                curr_step = state["step"]

                # 1. Momentum Update
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                
                if group["nesterov"]:
                    g_vec = g.add(buf, alpha=momentum)
                else:
                    g_vec = buf

                # 2. EF init
                if curr_step % skip_steps == 1:
                    state["cache_params"].copy_(p)
                    state["accum_vel"].zero_()

                # 3. Update
                if curr_step % skip_steps == 1:
                    u = zeropower_via_newtonschulz5(g_vec, steps=ns_steps)
                    norm_g = g_vec.norm() + 1e-16
                    norm_u = u.norm() + 1e-16
                    new_gain = norm_u / norm_g
                    state["spectral_gain"].copy_(new_gain)
                    update = u
                else:
                    gain = state["spectral_gain"]
                    update = g_vec * gain

                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)
                p.data.mul_(1 - lr * wd)
                p.data.add_(update, alpha=-adjusted_lr)

                # 5. EF accum
                state["accum_vel"].add_(g_vec)

                if curr_step % skip_steps == 0:
                    flatten_p = p.flatten()
                    flatten_cache = state["cache_params"].flatten()
                    dist = (flatten_p - flatten_cache).norm()
                    
                    beta2 = 0.95
                    state["EF_momentum_buffer"].lerp_(state["accum_vel"], 1 - beta2)
                    state["accum_vel"].lerp_(state["EF_momentum_buffer"], beta2)

                    u_ideal = zeropower_via_newtonschulz5(state["accum_vel"], steps=ns_steps)
                    p.data.copy_(state["cache_params"])
                    
                    scale = dist / (u_ideal.flatten().norm() + 1e-16)
                    p.data.add_(u_ideal.reshape(p.shape), alpha=-scale)

            # --- AdamW 组更新 (保持 KimiMuon 原样) ---
            adam_params_list = [p for p in group["params"] if not self.state[p].get("use_muon", False)]
            
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in adam_params_list:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                
                state["step"] += 1
                step = state["step"]
                
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g_adam = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g_adam, alpha=-lr / scale)

        return loss

