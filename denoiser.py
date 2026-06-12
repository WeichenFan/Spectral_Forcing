import torch
import torch.nn as nn
from model_jit import JiT_models


class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.pred_param = args.pred_param
        self.hybrid_x_weight_mode = args.hybrid_x_weight_mode
        self.hybrid_x_weight_power = args.hybrid_x_weight_power
        self.hybrid_x_weight_min = args.hybrid_x_weight_min
        self.hybrid_x_weight_max = args.hybrid_x_weight_max

        out_channels = 6 if self.pred_param == "hybrid" else 3
        self.net = JiT_models[args.model](
            input_size=args.img_size,
            window_size=getattr(args, "window_size", None),
            use_dct_patchify=not getattr(args, "disable_dct_patchify", False),
            use_dct_scale_schedule=not getattr(args, "disable_dct_scale_schedule", False),
            dct_fixed_scale=getattr(args, "dct_fixed_scale", 1.0),
            in_channels=3,
            out_channels=out_channels,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    def _hybrid_weight(self, t):
        if self.hybrid_x_weight_mode == "one_minus_t":
            w = (1.0 - t).clamp_min(self.t_eps)
        elif self.hybrid_x_weight_mode == "snr_inverse":
            s = t.clamp_min(self.t_eps)
            n = (1.0 - t).clamp_min(self.t_eps)
            snr = (s * s) / (n * n)
            w = 1.0 / (1.0 + snr)
        elif self.hybrid_x_weight_mode == "snr":
            s = t.clamp_min(self.t_eps)
            n = (1.0 - t).clamp_min(self.t_eps)
            snr = (s * s) / (n * n)
            w = snr / (1.0 + snr)
        else:
            raise ValueError(f"Unsupported hybrid_x_weight_mode: {self.hybrid_x_weight_mode}")

        w = w.pow(self.hybrid_x_weight_power)
        return w.clamp(self.hybrid_x_weight_min, self.hybrid_x_weight_max)

    def _pred_to_v(self, pred, z, t):
        denom = (1.0 - t).clamp_min(self.t_eps)

        if self.pred_param == "x":
            return (pred - z) / denom

        if self.pred_param == "v":
            return pred

        if self.pred_param == "hybrid":
            c = z.shape[1]
            if pred.shape[1] < 2 * c:
                raise ValueError(f"Hybrid mode expects at least {2 * c} channels, got {pred.shape[1]}")
            x_head = pred[:, :c]
            v_head = pred[:, c:2 * c]
            v_from_x = (x_head - z) / denom
            w = self._hybrid_weight(t)
            return w * v_from_x + (1.0 - w) * v_head

        raise ValueError(f"Unsupported pred_param: {self.pred_param}")

    def _run_net_to_v(self, z, t, labels):
        pred = self.net(z, t.flatten(), labels)
        return self._pred_to_v(pred, z, t)

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, labels):
        labels_dropped = self.drop_labels(labels) if self.training else labels

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        v_pred = self._run_net_to_v(z, t, labels_dropped)

        # l2 loss
        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()

        return loss

    @torch.no_grad()
    def generate(self, labels):
        device = labels.device
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, labels):
        # conditional
        v_cond = self._run_net_to_v(z, t, labels)

        # unconditional
        v_uncond = self._run_net_to_v(z, t, torch.full_like(labels, self.num_classes))

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(
            interval_mask,
            torch.full_like(t, self.cfg_scale),
            torch.ones_like(t)
        )

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        v_pred = self._forward_sample(z, t, labels)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        v_pred_t = self._forward_sample(z, t, labels)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
