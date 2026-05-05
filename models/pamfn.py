import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def smart_fn(fn, feat_list, func_list=False):
    if func_list is False:
        return [fn(x) for x in feat_list]
    return [fn[idx](x) for idx, x in enumerate(feat_list)]


def safe_torch_load(path, map_location="cpu", weights_only=None):
    kwargs = {"map_location": map_location}
    if weights_only is not None:
        try:
            obj = torch.load(path, weights_only=weights_only, **kwargs)
            patch_legacy_modules(obj)
            return obj
        except TypeError:
            pass
    obj = torch.load(path, **kwargs)
    patch_legacy_modules(obj)
    return obj


def patch_legacy_modules(obj):
    modules = []
    if isinstance(obj, nn.Module):
        modules.append(obj)
    elif isinstance(obj, dict):
        modules.extend([value for value in obj.values() if isinstance(value, nn.Module)])

    for module in modules:
        for submodule in module.modules():
            if isinstance(submodule, nn.GELU) and not hasattr(submodule, "approximate"):
                submodule.approximate = "none"
    return obj


class BaseConvBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            nn.GELU(),
        )

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class BaseModel(nn.Module):
    def __init__(self, in_dim, model_dim, drop_rate, modality="V"):
        super().__init__()
        dim = model_dim
        self.modality = modality
        self.embedding = nn.Sequential(
            nn.Conv1d(in_dim, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, dim, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True),
            nn.Dropout(0.3),
        )
        self.stage1 = BaseConvBlock(dim)
        self.stage2 = BaseConvBlock(dim)
        self.stage3 = BaseConvBlock(dim)
        self.pool = nn.AvgPool1d(2, 2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Conv1d(dim, 1, 1),
            nn.Sigmoid(),
        )
        self.mse = nn.MSELoss()

    def forward(self, feats):
        x = feats[self.modality]
        if len(x.shape) == 4:
            x = x.mean(dim=2)
        x = x.permute(0, 2, 1).contiguous()
        x = self.embedding(x)

        x1 = self.stage1(x)
        x1 = self.pool(x1)

        x2 = self.stage2(x1)
        x2 = self.pool(x2)

        x3 = self.stage3(x2)
        x3 = self.gap(x3)

        score = self.fc(x3).squeeze(dim=2)
        return score, {"feats": [x, x1, x2, x3]}

    def call_loss(self, pred, label, **kwargs):
        return self.mse(pred.squeeze(), label.squeeze())


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim_model, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        assert dim_model % num_heads == 0
        self.dim_head = dim_model // self.num_heads
        self.linear_layers = nn.ModuleList([nn.Linear(dim_model, dim_model) for _ in range(3)])
        self.drop = nn.Dropout(p=dropout)

    def self_attention(self, query, key, value, mask=None, dropout=None):
        scores = -1 * torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(query.size(-1))
        if mask is not None:
            mask = mask.unsqueeze(dim=1).repeat([1, self.num_heads, 1, 1])
            scores = scores + mask
        attn = torch.softmax(scores, dim=-1)
        if dropout is not None:
            attn = dropout(attn)
        context = torch.matmul(attn, value)
        return context

    def forward(self, input_q, input_k, input_v, mask=None):
        batch_size = input_v.size(0)
        q, k, v = [
            layer(x).view(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
            for layer, x in zip(self.linear_layers, (input_q, input_k, input_v))
        ]
        x = self.self_attention(q, k, v, mask=mask, dropout=self.drop)
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.dim_head)


class MSFusion(nn.Module):
    def __init__(self, dim, num_heads=1, dropout=0.1):
        super().__init__()
        self.fusion_s = MultiHeadAttention(num_heads=num_heads, dim_model=dim, dropout=dropout)
        self.proj = nn.Sequential(nn.Conv1d(dim, dim, 1))

    def forward(self, s, c):
        q = self.proj(c)
        q = q.permute(0, 2, 1).contiguous().view(-1, 1, q.shape[1])
        s = torch.stack(s, dim=2)
        s = s.permute(0, 3, 2, 1).contiguous().view(-1, s.shape[2], s.shape[1])
        ns = self.fusion_s(q, s, s)
        return ns.view(c.shape[0], c.shape[2], -1).permute(0, 2, 1).contiguous()


class CMFusion(nn.Module):
    def __init__(self, dim, k, num_heads=1, gap=False, dropout=0.1):
        super().__init__()
        self.k = k
        self.proj1 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(dim, dim, 1),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(True),
                    nn.Conv1d(dim, self.k, 1),
                )
                for _ in range(3)
            ]
        )
        self.ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(dim, dim, 3, 1, 1),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(True),
                    nn.Conv1d(dim, dim, 3, 1, 1),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(True),
                )
                for _ in range(self.k)
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1) if gap else nn.AvgPool1d(2, 2)
        self.fusion_f = MultiHeadAttention(num_heads=num_heads, dim_model=dim, dropout=dropout)
        self.proj2 = nn.Sequential(nn.Conv1d(2 * dim, dim, 1))
        self.policy = nn.Sequential(nn.Conv1d(3 * dim, k, 1))

    def gen_mask(self, fusion_kind):
        batch_size, k, t = fusion_kind.shape
        mask = torch.zeros_like(fusion_kind, device=fusion_kind.device)
        idx = torch.arange(0, k, device=fusion_kind.device).repeat([batch_size, 1, 1]).permute([0, 2, 1]).repeat([1, 1, t])
        pos_mask = idx > torch.mul(fusion_kind, idx).sum(dim=1, keepdim=True)
        mask[pos_mask] = -1e9
        return mask + fusion_kind - fusion_kind.detach()

    def gen_fusion_kind(self, f, tau):
        logit = self.pool(self.policy(torch.cat(f, dim=1)))
        fusion_kind = F.gumbel_softmax(logit, tau, hard=True, dim=1)
        return fusion_kind, self.gen_mask(fusion_kind)

    def forward(self, f, c, s, tau):
        att = torch.softmax(torch.stack(smart_fn(self.proj1, f, True), dim=1), dim=1)
        att = list(att.split(1, dim=2))
        nf = [torch.mul(_, torch.stack(f, dim=1)).sum(dim=1) for _ in att]
        nf = torch.stack([self.pool(self.ffn[idx](_) + _) for idx, _ in enumerate(nf)], dim=2)
        nf = nf.permute(0, 3, 2, 1).contiguous().view(-1, nf.shape[2], nf.shape[1])

        q = self.proj2(torch.cat([c, s], dim=1))
        q = q.permute(0, 2, 1).contiguous().view(-1, 1, q.shape[1])
        q = -1 * q
        fusion_kind, mask = self.gen_fusion_kind(f, tau)
        mask = mask.permute(0, 2, 1).contiguous().view(-1, 1, mask.shape[1])
        nf = self.fusion_f(q, nf, nf, mask=mask)
        return nf.view(c.shape[0], c.shape[2], -1).permute(0, 2, 1).contiguous(), fusion_kind


class FusionBlock(nn.Module):
    def __init__(self, dim, k, ms_heads, cm_heads, gap=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(dim * 3, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True),
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True),
        )
        self.short_cut = nn.Sequential(
            nn.Conv1d(dim * 3, dim, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True),
        )
        self.s_fusion = MSFusion(dim, ms_heads)
        self.f_fusion = CMFusion(dim, k, cm_heads, gap)

    def forward(self, c, f, s, tau):
        c = self.conv(c) + self.short_cut(c)
        s = self.s_fusion(s, c)
        f, action = self.f_fusion(f, c, s, tau)
        return torch.cat([c, s, f], dim=1), action


class PAMFN(nn.Module):
    def __init__(
        self,
        model_dim,
        fc_drop,
        fc_r,
        feat_drop,
        k,
        ms_heads,
        cm_heads,
        ckpt_dir,
        rgb_ckpt_name,
        flow_ckpt_name,
        audio_ckpt_name,
        dataset_name,
    ):
        super().__init__()
        self.model_r = safe_torch_load(osp.join(ckpt_dir, f"{dataset_name}_rgb_{rgb_ckpt_name}.pth"), map_location="cpu", weights_only=False)
        self.model_f = safe_torch_load(osp.join(ckpt_dir, f"{dataset_name}_flow_{flow_ckpt_name}.pth"), map_location="cpu", weights_only=False)
        self.model_a = safe_torch_load(osp.join(ckpt_dir, f"{dataset_name}_audio_{audio_ckpt_name}.pth"), map_location="cpu", weights_only=False)

        self.c = nn.Parameter(torch.zeros([model_dim * 3]), requires_grad=False)
        self.stage1 = FusionBlock(model_dim, k, ms_heads, cm_heads)
        self.stage2 = FusionBlock(model_dim, k, ms_heads, cm_heads)
        self.stage3 = FusionBlock(model_dim, k, ms_heads, cm_heads, True)
        self.pool = nn.AvgPool1d(2, 2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(feat_drop)

        self.fc = nn.Sequential(
            nn.Dropout(fc_drop),
            nn.Conv1d(model_dim * 3, model_dim // fc_r, 1),
            nn.BatchNorm1d(model_dim // fc_r),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(model_dim // fc_r, 1, 1),
            nn.Sigmoid(),
        )
        self.tau = nn.Parameter(torch.ones([]) * 10)
        self.mse = nn.MSELoss()

    def forward(self, input_feats):
        _, feat_r = self.model_r(input_feats)
        _, feat_a = self.model_a(input_feats)
        _, feat_f = self.model_f(input_feats)

        feats = [
            [self.drop(_.detach()) for _ in feat_r["feats"]],
            [self.drop(_.detach()) for _ in feat_f["feats"]],
            [self.drop(_) for _ in feat_a["feats"]],
        ]

        x = self.c.repeat([feats[0][0].shape[0], feats[0][0].shape[2] // 2, 1]).permute([0, 2, 1])
        x1, action1 = self.stage1(x, [_[0] for _ in feats], [_[1] for _ in feats], self.tau)
        x1 = self.pool(x1)
        x2, action2 = self.stage2(x1, [_[1] for _ in feats], [_[2] for _ in feats], self.tau)
        x2 = self.gap(x2)
        x3, action3 = self.stage3(x2, [_[2] for _ in feats], [_[3] for _ in feats], self.tau)
        score = self.fc(x3).squeeze(dim=2)
        return score, {"action": [action1, action2, action3]}


class PAMFNSyncBackbone(nn.Module):
    def __init__(
        self,
        model_dim,
        fc_drop,
        fc_r,
        feat_drop,
        k,
        ms_heads,
        cm_heads,
        ckpt_dir,
        rgb_ckpt_name,
        flow_ckpt_name,
        audio_ckpt_name,
        dataset_name,
        pamfn_ckpt_path=None,
    ):
        super().__init__()
        self.model_r = safe_torch_load(osp.join(ckpt_dir, f"{dataset_name}_rgb_{rgb_ckpt_name}.pth"), map_location="cpu", weights_only=False)
        self.model_f = safe_torch_load(osp.join(ckpt_dir, f"{dataset_name}_flow_{flow_ckpt_name}.pth"), map_location="cpu", weights_only=False)
        self.model_a = safe_torch_load(osp.join(ckpt_dir, f"{dataset_name}_audio_{audio_ckpt_name}.pth"), map_location="cpu", weights_only=False)

        self.model_dim = model_dim
        self.sync_out_dim = model_dim * 3
        self.c = nn.Parameter(torch.zeros([model_dim * 3]), requires_grad=False)
        self.stage1 = FusionBlock(model_dim, k, ms_heads, cm_heads)
        self.stage2 = FusionBlock(model_dim, k, ms_heads, cm_heads)
        self.stage3 = FusionBlock(model_dim, k, ms_heads, cm_heads, True)
        self.pool = nn.AvgPool1d(2, 2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(feat_drop)
        self.fc = nn.Sequential(
            nn.Dropout(fc_drop),
            nn.Conv1d(model_dim * 3, model_dim // fc_r, 1),
            nn.BatchNorm1d(model_dim // fc_r),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(model_dim // fc_r, 1, 1),
            nn.Sigmoid(),
        )
        self.tau = nn.Parameter(torch.ones([]) * 10)
        self.has_pamfn_init = False

        if pamfn_ckpt_path is not None:
            self.load_pamfn_weights(pamfn_ckpt_path)

    def load_pamfn_weights(self, ckpt_path):
        state = safe_torch_load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        missing, unexpected = self.load_state_dict(state, strict=False)
        self.has_pamfn_init = True
        return missing, unexpected

    @staticmethod
    def _set_requires_grad(module, enabled):
        for param in module.parameters():
            param.requires_grad = enabled

    def set_train_phase(self, phase):
        if phase == "warmup":
            self._set_requires_grad(self.model_r, False)
            self._set_requires_grad(self.model_f, False)
            self._set_requires_grad(self.model_a, False)
            freeze_fusion = self.has_pamfn_init
            self._set_requires_grad(self.stage1, not freeze_fusion)
            self._set_requires_grad(self.stage2, not freeze_fusion)
            self._set_requires_grad(self.stage3, not freeze_fusion)
            self._set_requires_grad(self.fc, True)
            self.tau.requires_grad = not freeze_fusion
        else:
            self._set_requires_grad(self.model_r, False)
            self._set_requires_grad(self.model_f, False)
            self._set_requires_grad(self.model_a, True)
            self._set_requires_grad(self.stage1, True)
            self._set_requires_grad(self.stage2, True)
            self._set_requires_grad(self.stage3, True)
            self._set_requires_grad(self.fc, True)
            self.tau.requires_grad = True

    def iter_pretrained_parameters(self):
        modules = [self.model_r, self.model_f, self.model_a]
        if self.has_pamfn_init:
            modules.extend([self.stage1, self.stage2, self.stage3, self.fc])
        for module in modules:
            for param in module.parameters():
                yield param
        if self.has_pamfn_init:
            yield self.tau

    def forward(self, input_feats):
        _, feat_r = self.model_r(input_feats)
        _, feat_a = self.model_a(input_feats)
        _, feat_f = self.model_f(input_feats)

        feats = [
            [self.drop(_.detach()) for _ in feat_r["feats"]],
            [self.drop(_.detach()) for _ in feat_f["feats"]],
            [self.drop(_) for _ in feat_a["feats"]],
        ]

        x = self.c.repeat([feats[0][0].shape[0], feats[0][0].shape[2] // 2, 1]).permute([0, 2, 1])
        x1, action1 = self.stage1(x, [_[0] for _ in feats], [_[1] for _ in feats], self.tau)
        x1_pool = self.pool(x1)

        x2, action2 = self.stage2(x1_pool, [_[1] for _ in feats], [_[2] for _ in feats], self.tau)
        sync_tokens = x2.transpose(1, 2).contiguous()

        x2_gap = self.gap(x2)
        x3, action3 = self.stage3(x2_gap, [_[2] for _ in feats], [_[3] for _ in feats], self.tau)
        pooled_tokens = x3.transpose(1, 2).contiguous()
        aux_score = self.fc(x3).squeeze(dim=2)

        return {
            "sync_tokens": sync_tokens,
            "pooled_tokens": pooled_tokens,
            "aux_score": aux_score,
            "action": [action1, action2, action3],
        }
