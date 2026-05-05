import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from joint_fisv.data import build_joint_datasets
from joint_fisv.engine import evaluate, train
from joint_fisv.losses import JointLoss
from joint_fisv.model import DualPrototypeAVLikert
from models.pamfn import safe_torch_load

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but not available.\n"
                "python: {}\n"
                "torch: {}\n"
                "torch.version.cuda: {}\n"
                "Hint: you may be running the CPU-only interpreter instead of the 'gdlt-win' environment.\n"
                "Try: conda run -n gdlt-win python joint_fisv_main.py ...".format(
                    sys.executable,
                    torch.__version__,
                    torch.version.cuda,
                )
            )
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    raise ValueError("Unsupported device '{}'".format(device_arg))


def ensure_path(path, description):
    if path and not os.path.exists(path):
        raise FileNotFoundError("{} not found: {}".format(description, path))


def build_optimizer(model, args):
    pretrained_ids = {id(param) for param in model.iter_pretrained_parameters()}
    base_params = []
    pretrained_params = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in pretrained_ids:
            pretrained_params.append(param)
        else:
            base_params.append(param)

    param_groups = []
    if base_params:
        param_groups.append({"params": base_params, "lr": args.lr})
    if pretrained_params:
        param_groups.append({"params": pretrained_params, "lr": args.lr * args.pretrained_lr_scale})

    if args.optim == "sgd":
        return torch.optim.SGD(param_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.optim == "adam":
        return torch.optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    if args.optim == "adamw":
        return torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    raise ValueError("Unsupported optimizer '{}'".format(args.optim))


def build_scheduler(optimizer, args):
    if args.lr_decay == "none":
        return None
    if args.lr_decay == "cos":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=args.lr * args.decay_rate)
    if args.lr_decay == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[max(args.epoch - 30, 1)], gamma=args.decay_rate)
    raise ValueError("Unsupported lr decay '{}'".format(args.lr_decay))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-feat-dir", type=str, required=True)
    parser.add_argument("--mm-feat-dir", type=str, required=True)
    parser.add_argument("--joint-train-label", type=str, required=True)
    parser.add_argument("--joint-test-label", type=str, required=True)
    parser.add_argument("--clip-num", type=int, default=124)
    parser.add_argument("--tes-score-max", type=float, default=None)
    parser.add_argument("--pcs-score-max", type=float, default=None)
    parser.add_argument("--rgb-feat-name", type=str, default="VST")
    parser.add_argument("--flow-feat-name", type=str, default="I3D")
    parser.add_argument("--audio-feat-name", type=str, default="AST")

    parser.add_argument("--visual-in-dim", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-head", type=int, default=1)
    parser.add_argument("--n-encoder", type=int, default=1)
    parser.add_argument("--n-decoder", type=int, default=2)
    parser.add_argument("--n-query-tes", type=int, default=4)
    parser.add_argument("--n-query-pcs", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.7)
    parser.add_argument("--no-shared-prototype", action="store_true")
    parser.add_argument("--pcs-memory", type=str, default="tokens", choices=["tokens", "pooled"])

    parser.add_argument("--pamfn-model-dim", type=int, default=256)
    parser.add_argument("--pamfn-fc-drop", type=float, default=0.0)
    parser.add_argument("--pamfn-fc-r", type=int, default=2)
    parser.add_argument("--pamfn-feat-drop", type=float, default=0.5)
    parser.add_argument("--pamfn-k", type=int, default=6)
    parser.add_argument("--pamfn-ms-heads", type=int, default=1)
    parser.add_argument("--pamfn-cm-heads", type=int, default=1)
    parser.add_argument("--pamfn-ckpt-dir", type=str, default="./pretrained_models/feats1")
    parser.add_argument("--pamfn-rgb-ckpt-name", type=str, default="VST")
    parser.add_argument("--pamfn-flow-ckpt-name", type=str, default="I3D")
    parser.add_argument("--pamfn-audio-ckpt-name", type=str, default="AST")
    parser.add_argument("--pamfn-dataset-name", type=str, default="PCS")
    parser.add_argument("--pamfn-ckpt", type=str, default=None)

    parser.add_argument("--epoch", type=int, default=160)
    parser.add_argument("--warmup-epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--pretrained-lr-scale", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optim", type=str, default="adamw", choices=["sgd", "adam", "adamw"])
    parser.add_argument("--lr-decay", type=str, default="cos", choices=["none", "cos", "multistep"])
    parser.add_argument("--decay-rate", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--lambda-rank", type=float, default=0.3)
    parser.add_argument("--lambda-proto", type=float, default=0.5)
    parser.add_argument("--lambda-gap", type=float, default=0.2)
    parser.add_argument("--lambda-aux", type=float, default=0.5)
    parser.add_argument("--rank-threshold", type=float, default=0.05)
    parser.add_argument("--gap-beta", type=float, default=5.0)
    parser.add_argument("--triplet-margin", type=float, default=1.0)
    parser.add_argument("--pcs-reg-weight", type=float, default=1.2)

    parser.add_argument("--ckpt-path", type=str, default="./ckpt/joint_fisv_best.pkl")
    parser.add_argument("--error-report-path", type=str, default="./logs/joint_fisv_error_report.txt")
    parser.add_argument("--log-dir", type=str, default="./logs/joint_fisv")
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    return parser.parse_args()


def build_model(args):
    return DualPrototypeAVLikert(
        visual_in_dim=args.visual_in_dim,
        hidden_dim=args.hidden_dim,
        n_head=args.n_head,
        n_encoder=args.n_encoder,
        n_decoder=args.n_decoder,
        n_query_tes=args.n_query_tes,
        n_query_pcs=args.n_query_pcs,
        dropout=args.dropout,
        pamfn_model_dim=args.pamfn_model_dim,
        pamfn_fc_drop=args.pamfn_fc_drop,
        pamfn_fc_r=args.pamfn_fc_r,
        pamfn_feat_drop=args.pamfn_feat_drop,
        pamfn_k=args.pamfn_k,
        pamfn_ms_heads=args.pamfn_ms_heads,
        pamfn_cm_heads=args.pamfn_cm_heads,
        pamfn_ckpt_dir=args.pamfn_ckpt_dir,
        pamfn_rgb_ckpt_name=args.pamfn_rgb_ckpt_name,
        pamfn_flow_ckpt_name=args.pamfn_flow_ckpt_name,
        pamfn_audio_ckpt_name=args.pamfn_audio_ckpt_name,
        pamfn_dataset_name=args.pamfn_dataset_name,
        pamfn_ckpt_path=args.pamfn_ckpt,
        shared_prototype=not args.no_shared_prototype,
        pcs_memory=args.pcs_memory,
    )


def load_checkpoint_if_needed(model, args, device):
    ckpt_path = args.ckpt if args.ckpt is not None else args.ckpt_path
    ensure_path(ckpt_path, "Checkpoint")
    state = safe_torch_load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
        return state
    model.load_state_dict(state)
    return {}


if __name__ == "__main__":
    args = parse_args()
    setup_seed(args.seed)
    device = resolve_device(args.device)

    ensure_path(args.visual_feat_dir, "Visual feature directory")
    ensure_path(args.mm_feat_dir, "Multimodal feature directory")
    ensure_path(args.joint_train_label, "Joint training label file")
    ensure_path(args.joint_test_label, "Joint test label file")
    ensure_path(args.pamfn_ckpt_dir, "PAMFN checkpoint directory")
    if args.pamfn_ckpt is not None:
        ensure_path(args.pamfn_ckpt, "PAMFN checkpoint")

    train_dataset, test_dataset = build_joint_datasets(
        visual_feat_dir=args.visual_feat_dir,
        mm_feat_dir=args.mm_feat_dir,
        train_label_path=args.joint_train_label,
        test_label_path=args.joint_test_label,
        clip_num=args.clip_num,
        tes_score_max=args.tes_score_max,
        pcs_score_max=args.pcs_score_max,
        rgb_feat_name=args.rgb_feat_name,
        flow_feat_name=args.flow_feat_name,
        audio_feat_name=args.audio_feat_name,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args).to(device)
    criterion = JointLoss(
        lambda_rank=args.lambda_rank,
        lambda_proto=args.lambda_proto,
        lambda_gap=args.lambda_gap,
        lambda_aux=args.lambda_aux,
        rank_threshold=args.rank_threshold,
        gap_beta=args.gap_beta,
        triplet_margin=args.triplet_margin,
        pcs_reg_weight=args.pcs_reg_weight,
    )

    writer = SummaryWriter(args.log_dir) if SummaryWriter is not None and not args.test else None

    if args.test:
        load_checkpoint_if_needed(model, args, device)
        metrics = evaluate(model, criterion, test_loader, device, args, writer=writer)
        print("Eval mean SRCC: {:.4f} | TES {:.4f} | PCS {:.4f}".format(metrics["mean_srcc"], metrics["tes_srcc"], metrics["pcs_srcc"]))
    else:
        optimizer = build_optimizer(model, args)
        scheduler = build_scheduler(optimizer, args)
        result = train(model, criterion, train_loader, test_loader, optimizer, scheduler, device, args, writer=writer)
        if result["best_metrics"] is not None:
            best = result["best_metrics"]
            print(
                "Best epoch {} | mean SRCC {:.4f} | TES {:.4f} | PCS {:.4f}".format(
                    result["best_epoch"], best["mean_srcc"], best["tes_srcc"], best["pcs_srcc"]
                )
            )

    if writer is not None:
        writer.close()
