import torch
import torch.nn as nn
import torch.nn.functional as F

from models.triplet_loss import HardTripletLoss


def pairwise_rank_loss(pred, label, threshold):
    label_diff = label.unsqueeze(1) - label.unsqueeze(0)
    mask = label_diff.abs() > threshold
    if not mask.any():
        return pred.new_tensor(0.0)
    pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
    sign = label_diff.sign()
    loss = F.relu(-sign * pred_diff)
    return loss[mask].mean()


def gap_aware_consistency_loss(tes_embed, pcs_embed, tes_label, pcs_label, beta):
    tes_repr = F.normalize(tes_embed.mean(dim=1), dim=-1)
    pcs_repr = F.normalize(pcs_embed.mean(dim=1), dim=-1)
    cosine_gap = 1 - (tes_repr * pcs_repr).sum(dim=-1)
    weight = torch.exp(-beta * (tes_label - pcs_label).abs())
    return (weight * cosine_gap).mean()


class JointLoss(nn.Module):
    def __init__(
        self,
        lambda_rank=0.3,
        lambda_proto=0.5,
        lambda_gap=0.2,
        lambda_aux=0.5,
        rank_threshold=0.05,
        gap_beta=5.0,
        triplet_margin=1.0,
        pcs_reg_weight=1.0,
    ):
        super().__init__()
        self.mse = nn.MSELoss()
        self.triplet = HardTripletLoss(margin=triplet_margin, hardest=True)
        self.lambda_rank = lambda_rank
        self.lambda_proto = lambda_proto
        self.lambda_gap = lambda_gap
        self.lambda_aux = lambda_aux
        self.rank_threshold = rank_threshold
        self.gap_beta = gap_beta
        self.pcs_reg_weight = pcs_reg_weight

    def _prototype_loss(self, embed):
        batch_size, n_query, dim = embed.shape
        labels = torch.arange(n_query, device=embed.device).repeat(batch_size)
        return self.triplet(embed.reshape(-1, dim), labels)

    def forward(self, outputs, batch, aux_decay=1.0):
        tes_label = batch["tes_label"]
        pcs_label = batch["pcs_label"]

        tes_reg = self.mse(outputs["tes_score"], tes_label)
        pcs_reg = self.mse(outputs["pcs_score"], pcs_label)
        reg_loss = tes_reg + self.pcs_reg_weight * pcs_reg

        rank_loss = pairwise_rank_loss(outputs["tes_score"], tes_label, self.rank_threshold)
        rank_loss = rank_loss + pairwise_rank_loss(outputs["pcs_score"], pcs_label, self.rank_threshold)

        proto_loss = self._prototype_loss(outputs["tes_embed"]) + self._prototype_loss(outputs["pcs_embed"])
        gap_loss = gap_aware_consistency_loss(outputs["tes_embed"], outputs["pcs_embed"], tes_label, pcs_label, self.gap_beta)
        aux_loss = self.mse(outputs["aux_pcs_score"], pcs_label)
        aux_weight = self.lambda_aux * max(aux_decay, 0.0)

        total = reg_loss
        total = total + self.lambda_rank * rank_loss
        total = total + self.lambda_proto * proto_loss
        total = total + self.lambda_gap * gap_loss
        total = total + aux_weight * aux_loss

        stats = {
            "loss": total.detach(),
            "reg": reg_loss.detach(),
            "tes_reg": tes_reg.detach(),
            "pcs_reg": pcs_reg.detach(),
            "rank": rank_loss.detach(),
            "proto": proto_loss.detach(),
            "gap": gap_loss.detach(),
            "aux": aux_loss.detach(),
            "aux_weight": torch.tensor(aux_weight, device=total.device),
        }
        return total, stats
