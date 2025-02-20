import time
import math
import torch
import torch.nn as nn

from quant_utils import *

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class Aespa:
    def __init__(self, layer):
        self.layer = layer        
        self.quantizer = None
        self.H = None
        self.cov_G = None

    def compute_cov_in_batch(self, _, inp, out):
        if self.H is None:
            self.H = 0
            self.n_data_in = 0

        inp = inp[0].data
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))  # shape = [BL, d]
        inp = inp.t()  # shape = [d, BL]

        n_current = inp.shape[-1]
        self.H *= self.n_data_in / (self.n_data_in + n_current)
        self.n_data_in += n_current
        inp = math.sqrt(2 / self.n_data_in) * inp.float()
        self.H += inp.matmul(inp.t())


    def compute_cov_out_batch(self, _, inps, outs, n_heads):
        if not hasattr(self, "cov_out"):
            self.cov_out = 0
            self.n_data_out = 0
        
        head_dim = outs.shape[-1] // n_heads
        outs = outs.view(outs.shape[0], outs.shape[1], n_heads, head_dim).transpose(1, 2).contiguous()  # [B, H, L, d_h]
        outs = outs.transpose(0, 1).view(n_heads, -1, head_dim).transpose(-1, -2).contiguous()  # [H, d_h, BL]

        n_current = outs.shape[-1]
        self.cov_out *= self.n_data_out / (self.n_data_out + n_current)
        self.n_data_out += n_current
        outs = math.sqrt(2 / self.n_data_out) * outs.float()
        self.cov_out += outs @ outs.transpose(-1, -2)


    def refine_quant_params(self, use_zfold: bool, hyperparams: dict):
        assert self.quantizer is not None, "Quantizer should be defined first."
        assert self.H is not None, "Hessian should be computed first."

        W = self.layer.weight.data.clone()
        if not self.quantizer.ready():
            self.quantizer.find_params(W)
        
        W, H = W.float(), self.H.clone()
        W, H = filter_dead_neuron(W, H, replace=hyperparams['replace'], percdamp=hyperparams['percdamp'], apply_damping=True)
        
        tick = time.time()
        if use_zfold:
            return refine_qparams_zfold({self.name: self}, [self.name], hyperparams)
        else:
            scale, zero, zeta = self.quantizer.scale.view([-1, 1]), self.quantizer.zero.view([-1, 1]), self.quantizer.zeta.view([1, -1])
            n_bits = self.quantizer.nbits

            # compute initial loss perturbation incurred by quantization
            loss_perturb_before = compute_loss_perturb(W, scale, zero, zeta, n_bits, H)

            # update scale and zero-point
            scale, zero = find_quant_params(W, zeta, n_bits, self.quantizer.sym, H)

            # compute loss perturbation after update
            loss_perturb_after = compute_loss_perturb(W, scale, zero, zeta, n_bits, H)

            delta_loss_improvement = loss_perturb_before - loss_perturb_after
            
            self.quantizer.scale.data = scale.view(self.quantizer.scale.shape)
            self.quantizer.zero.data = zero.view(self.quantizer.zero.shape)

            return delta_loss_improvement, 1, time.time() - tick

    def quant(self, opts:dict, hyperparams: dict):
        assert self.quantizer is not None, "Quantizer should be defined first."
        assert self.H is not None, "Hessian should be computed first."

        W = self.layer.weight.data.clone()
        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)
        orig_w_shape, orig_w_dtype = W.shape, W.dtype
        W = W.float()

        # Quant. Params.
        scale, zero, zeta = self.quantizer.scale.view([-1, 1]), self.quantizer.zero.view([-1, 1]), self.quantizer.zeta.view([1, -1])
        n_bits = self.quantizer.nbits

        H, cov_G = self.H.clone(), self.cov_G.clone() if self.cov_G is not None else None

        # pre-processing
        W, H = filter_dead_neuron(W, H, replace=hyperparams['replace'], percdamp=hyperparams['percdamp'], apply_damping=True)
        if len(H.shape) == 2:  # common Hessian for all heads
            H = H.unsqueeze(0)
        num_heads = H.shape[0] if cov_G is None else cov_G.shape[0]
        hidden_size = W.shape[-1]
        head_dim = W.shape[0] // num_heads
        W = W.view(num_heads, head_dim, hidden_size)
        scale, zero = scale.view(num_heads, head_dim, 1), zero.view(num_heads, head_dim, 1)
        zeta = zeta.view([1, 1, -1])

        # initialize weight-rounding policy
        if opts['optq_init']:
            W_update = self.optq(W, H, scale, zero, zeta, n_bits, act_order=opts['act_order'])
            if not opts['learn_rounding']:
                print(f'|{self.i}: {self.name : <24}|GPU memory: {torch.cuda.max_memory_allocated("cuda") / 1024**3:.3f}\t|')
        else:
            W_update = W
        
        # weight-rounding optimization via learning
        if opts['learn_rounding']:
            Q = self.adaround(W, W_update, H, cov_G, scale, zero, zeta, n_bits, opts)
        else:
            Q = quantize_zfold(W_update, scale, zero, zeta, n_bits)

        # assign quantized (fake-quant) weights
        self.layer.weight.data = Q.reshape(orig_w_shape).to(orig_w_dtype)


    def optq(self, W, H, scale, zero, zeta, n_bits, act_order):
        W, H = W.clone(), H.clone()
        n_columns = W.shape[-1]

        if act_order:
            num_heads, hidden_size = W.shape[0], H.shape[-1]
            if H.shape[0] == 1:  # Common Hessian for all heads
                W = W.view(1, -1, hidden_size)
            perm_multi_head = torch.zeros((H.shape[0], hidden_size), dtype=torch.int64, device=H.device)
            invperm_multi_head = torch.zeros_like(perm_multi_head)
            zeta_multi_head = torch.zeros((H.shape[0], *zeta.shape[1:]), device=zeta.device)
            for idx_head in range(H.shape[0]):
                perm = torch.argsort(torch.diag(H[idx_head]), descending=True)
                invperm_multi_head[idx_head] = torch.argsort(perm)
                W[idx_head] = W[idx_head][:, perm]
                H[idx_head] = H[idx_head][perm][:, perm]
                zeta_multi_head[idx_head] = zeta[0][:, perm]
                perm_multi_head[idx_head] = perm

            W = W.view(num_heads, -1, hidden_size)
            zeta = zeta_multi_head

        # Cholesky Decomposition
        U = torch.zeros_like(H)
        for idx_head in range(H.shape[0]):
            U[idx_head] = torch.linalg.cholesky(
                torch.cholesky_inverse(torch.linalg.cholesky(H[idx_head])), upper=True
            )
        
        W_update = torch.zeros_like(W)
        blocksize = 128
        for i1 in range(0, n_columns, blocksize):
            i2 = min(i1 + blocksize, n_columns)
            count = i2 - i1

            W1 = W[..., i1:i2].clone()
            Err1 = torch.zeros_like(W1)
            U1 = U[..., i1:i2, i1:i2]

            for i in range(count):
                W_update[..., i1+i] = W1[..., i]
                
                q = quantize_zfold(W1[..., i].unsqueeze(-1), scale, zero, zeta[..., i1+i].unsqueeze(-1), n_bits).squeeze(-1)
                err1 = (W1[..., i] - q) / U1[..., i, i].unsqueeze(-1)
                W1[..., i:] -= torch.matmul(err1.unsqueeze(-1), U1[..., i, i:].unsqueeze(-2))
                Err1[..., i] = err1
            W[..., i2:] -= torch.matmul(Err1, U[..., i1:i2, i2:])
        
        if act_order:
            if H.shape[0] == 1:
                W_update = W_update.view(1, -1, hidden_size)
            for idx_head in range(H.shape[0]):
                W_update[idx_head] = W_update[idx_head][:, invperm_multi_head[idx_head]]
            W_update = W_update.view(num_heads, -1, hidden_size)

        return W_update

    def adaround(self, W_org, W_update, H, cov_G, scale, zero, zeta, n_bits, opts: dict):
        lr, num_iters = opts['lr'], opts['num_iters']
        round_weight = opts['round_weight_qkv'] if self.name in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"] else opts['round_weight']

        scale = scale * zeta

        print_period = int(num_iters * 0.2)
        with torch.enable_grad():
            sigm = RectifiedSigmoid(-0.1, 1.1)
            sb = nn.Parameter(sigm.inverse(W_update / scale - torch.floor(W_update / scale)))
            optimizer = torch.optim.Adam([sb], lr=lr)

            round_loss_func = RoundLoss(max_count=num_iters, b_range=(20, 2), decay_start=0.0, warmup=0.2, p_norm=2.0)

            for i in range(num_iters):
                q = torch.clamp(torch.floor(W_update / scale) + sigm(sb) + zero, 0, 2**n_bits-1)
                q = scale * (q - zero)
                e = q - W_org
                recon_loss = ((e @ H) * e).sum() if cov_G is None else (cov_G * (e @ H @ e.transpose(-1, -2))).sum()
                round_loss = round_loss_func(i, sigm(sb))
                total_loss = recon_loss + round_weight * round_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                if i == 0 or (i + 1) % print_period == 0:
                    if self.i is None:
                        print(f'|{self.name : <27}| {i+1: <2}\t| {float(recon_loss):.3f}\t| {float(round_loss):.3f}\t| {torch.cuda.max_memory_allocated("cuda") / 1024**3: .3f}\t|')
                    else:
                        print(f'|{self.i}: {self.name : <24}| {i+1: <2}\t| {float(recon_loss):.3f}\t| {float(round_loss):.3f}\t|{torch.cuda.max_memory_allocated("cuda") / 1024**3: .3f}\t|')
            print('+===========================+================+=================+=================+')

        Q = torch.clamp(torch.floor(W_update / scale) + (sb >= 0).float() + zero, 0, 2**n_bits-1)
        Q = scale * (Q - zero)

        return Q

    def free(self):
        self.H = None
        self.cov_G = None

        torch.cuda.empty_cache()


class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay, start_b, end_b):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t-self.start_decay) / (self.t_max-self.start_decay)
            return self.end_b + (self.start_b-self.end_b)*max(0.0, 1 - rel_t)
        
class RoundLoss(nn.Module):
    def __init__(self, max_count, b_range, decay_start, warmup, p_norm):
        super(RoundLoss, self).__init__()
        self.loss_start = max_count * warmup
        # NOTE: cosine temp decay does not improve accuracy.
        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1-warmup)*decay_start, start_b=b_range[0], end_b=b_range[1])
        self.p_norm = p_norm
        self.b = 0

    def forward(self, iter_count, sb):
        """Compute regularization term to optimize the rounding policy"""
        if iter_count < self.loss_start:
            return 0
        else:
            self.b = self.temp_decay(iter_count)
            return (1 - (2*sb - 1).abs().pow(self.b)).sum()
        
class RectifiedSigmoid(nn.Module):
    """
    Implementation of Rectified Sigmoid Function
    Based on https://arxiv.org/pdf/1712.01312
    """

    def __init__(self, gamma, zeta):
        super(RectifiedSigmoid, self).__init__()
        self.gamma = gamma
        self.zeta = zeta

    def forward(self, x):
        return torch.clamp(torch.sigmoid(x)*(self.zeta-self.gamma) + self.gamma, 0, 1)

    def inverse(self, y):
        """return x that satisfies y = RectifiedSigmoid(x)"""
        return -torch.log((self.zeta-self.gamma)/(y-self.gamma) - 1)