import torch
import torch.nn as nn

from models.pamfn import PAMFNSyncBackbone
from models.transformer import Transformer


class VisualMemoryEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_head, n_encoder, dropout):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Conv1d(in_dim, in_dim // 2, kernel_size=1),
            nn.BatchNorm1d(in_dim // 2),
            nn.ReLU(),
            nn.Conv1d(in_dim // 2, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
        )
        self.encoder = Transformer(
            d_model=hidden_dim,
            nhead=n_head,
            num_encoder_layers=n_encoder,
            num_decoder_layers=1,
            dim_feedforward=3 * hidden_dim,
            batch_first=True,
            dropout=dropout,
        ).encoder

    def forward(self, x):
        x = self.in_proj(x.transpose(1, 2)).transpose(1, 2)
        return self.encoder(x)


class PrototypeBank(nn.Module):
    def __init__(self, hidden_dim, n_query_tes, n_query_pcs, shared=True):
        super().__init__()
        self.shared = shared
        self.n_query_tes = n_query_tes
        self.n_query_pcs = n_query_pcs
        if shared:
            if n_query_tes != n_query_pcs:
                raise ValueError("Shared prototype mode requires n_query_tes == n_query_pcs.")
            self.base = nn.Embedding(n_query_tes, hidden_dim)
            self.delta_tes = nn.Parameter(torch.zeros(n_query_tes, hidden_dim))
            self.delta_pcs = nn.Parameter(torch.zeros(n_query_pcs, hidden_dim))
            nn.init.xavier_uniform_(self.base.weight)
        else:
            self.tes = nn.Embedding(n_query_tes, hidden_dim)
            self.pcs = nn.Embedding(n_query_pcs, hidden_dim)
            nn.init.xavier_uniform_(self.tes.weight)
            nn.init.xavier_uniform_(self.pcs.weight)

    def forward(self):
        if self.shared:
            return self.base.weight + self.delta_tes, self.base.weight + self.delta_pcs
        return self.tes.weight, self.pcs.weight


class LikertHead(nn.Module):
    def __init__(self, hidden_dim, n_head, n_decoder, n_query, dropout):
        super().__init__()
        self.decoder = Transformer(
            d_model=hidden_dim,
            nhead=n_head,
            num_encoder_layers=1,
            num_decoder_layers=n_decoder,
            dim_feedforward=3 * hidden_dim,
            batch_first=True,
            dropout=dropout,
        ).decoder
        self.regressor = nn.Linear(hidden_dim, n_query)
        self.register_buffer("weight", torch.linspace(0, 1, n_query))

    def forward(self, memory, prototype):
        batch_size = memory.shape[0]
        query = prototype.unsqueeze(0).expand(batch_size, -1, -1)
        embed = self.decoder(query, memory)
        logits = torch.diagonal(self.regressor(embed), dim1=-2, dim2=-1)
        dist = torch.sigmoid(logits)
        dist = dist / (dist.sum(dim=1, keepdim=True) + 1e-6)
        score = torch.sum(self.weight.unsqueeze(0) * dist, dim=1)
        return score, embed, dist


class DualPrototypeAVLikert(nn.Module):
    def __init__(
        self,
        visual_in_dim=1024,
        hidden_dim=256,
        n_head=1,
        n_encoder=1,
        n_decoder=2,
        n_query_tes=4,
        n_query_pcs=4,
        dropout=0.7,
        pamfn_model_dim=256,
        pamfn_fc_drop=0.0,
        pamfn_fc_r=2,
        pamfn_feat_drop=0.5,
        pamfn_k=6,
        pamfn_ms_heads=1,
        pamfn_cm_heads=1,
        pamfn_ckpt_dir="./pretrained_models/feats1",
        pamfn_rgb_ckpt_name="VST",
        pamfn_flow_ckpt_name="I3D",
        pamfn_audio_ckpt_name="AST",
        pamfn_dataset_name="PCS",
        pamfn_ckpt_path=None,
        shared_prototype=True,
        pcs_memory="tokens",
    ):
        super().__init__()
        self.visual_encoder = VisualMemoryEncoder(visual_in_dim, hidden_dim, n_head, n_encoder, dropout)
        self.sync_encoder = PAMFNSyncBackbone(
            model_dim=pamfn_model_dim,
            fc_drop=pamfn_fc_drop,
            fc_r=pamfn_fc_r,
            feat_drop=pamfn_feat_drop,
            k=pamfn_k,
            ms_heads=pamfn_ms_heads,
            cm_heads=pamfn_cm_heads,
            ckpt_dir=pamfn_ckpt_dir,
            rgb_ckpt_name=pamfn_rgb_ckpt_name,
            flow_ckpt_name=pamfn_flow_ckpt_name,
            audio_ckpt_name=pamfn_audio_ckpt_name,
            dataset_name=pamfn_dataset_name,
            pamfn_ckpt_path=pamfn_ckpt_path,
        )
        self.sync_adapter = nn.Sequential(
            nn.Conv1d(self.sync_encoder.sync_out_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
        )
        self.prototype_bank = PrototypeBank(hidden_dim, n_query_tes, n_query_pcs, shared=shared_prototype)
        self.tes_head = LikertHead(hidden_dim, n_head, n_decoder, n_query_tes, dropout)
        self.pcs_head = LikertHead(hidden_dim, n_head, n_decoder, n_query_pcs, dropout)
        self.pcs_memory = pcs_memory

    def set_train_phase(self, phase):
        self.sync_encoder.set_train_phase(phase)

    def iter_pretrained_parameters(self):
        return self.sync_encoder.iter_pretrained_parameters()

    def forward(self, batch):
        visual_memory = self.visual_encoder(batch["visual_seq"])
        sync_out = self.sync_encoder(
            {"V": batch["rgb_seq"], "F": batch["flow_seq"], "A": batch["audio_seq"]}
        )
        sync_memory = sync_out["sync_tokens"] if self.pcs_memory == "tokens" else sync_out["pooled_tokens"]
        sync_memory = self.sync_adapter(sync_memory.transpose(1, 2)).transpose(1, 2)

        tes_proto, pcs_proto = self.prototype_bank()
        tes_score, tes_embed, tes_dist = self.tes_head(visual_memory, tes_proto)
        pcs_score, pcs_embed, pcs_dist = self.pcs_head(sync_memory, pcs_proto)

        return {
            "tes_score": tes_score,
            "pcs_score": pcs_score,
            "tes_embed": tes_embed,
            "pcs_embed": pcs_embed,
            "tes_dist": tes_dist,
            "pcs_dist": pcs_dist,
            "sync_tokens": sync_out["sync_tokens"],
            "pcs_memory": sync_memory,
            "aux_pcs_score": sync_out["aux_score"].squeeze(-1),
        }
