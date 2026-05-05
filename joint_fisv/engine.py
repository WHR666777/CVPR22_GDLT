import os

import numpy as np
import torch
from scipy.stats import spearmanr


class Meter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value, n):
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self):
        if self.count == 0:
            return 0.0
        return self.sum / self.count


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def safe_srcc(preds, labels):
    coef, _ = spearmanr(preds, labels)
    if coef != coef:
        return 0.0
    return float(coef)


def aux_decay(epoch, warmup_epochs):
    if warmup_epochs <= 0:
        return 0.0
    return max(0.0, 1.0 - (epoch / float(warmup_epochs)))


def collect_predictions(storage, outputs, batch):
    storage["tes_pred"].extend(outputs["tes_score"].detach().cpu().numpy().tolist())
    storage["pcs_pred"].extend(outputs["pcs_score"].detach().cpu().numpy().tolist())
    storage["tes_label"].extend(batch["tes_label"].detach().cpu().numpy().tolist())
    storage["pcs_label"].extend(batch["pcs_label"].detach().cpu().numpy().tolist())
    storage["video_id"].extend(batch["video_id"])


def compute_metrics(storage):
    tes_srcc = safe_srcc(storage["tes_pred"], storage["tes_label"])
    pcs_srcc = safe_srcc(storage["pcs_pred"], storage["pcs_label"])
    return {
        "tes_srcc": tes_srcc,
        "pcs_srcc": pcs_srcc,
        "mean_srcc": 0.5 * (tes_srcc + pcs_srcc),
    }


def run_epoch(model, criterion, dataloader, device, optimizer=None, epoch=0, warmup_epochs=0):
    training = optimizer is not None
    phase = "warmup" if epoch < warmup_epochs else "joint"
    model.set_train_phase(phase)
    model.train(training)

    meter_names = ["loss", "reg", "tes_reg", "pcs_reg", "rank", "proto", "gap", "aux"]
    meters = {name: Meter() for name in meter_names}
    storage = {"tes_pred": [], "pcs_pred": [], "tes_label": [], "pcs_label": [], "video_id": []}

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)
            loss, stats = criterion(outputs, batch, aux_decay=aux_decay(epoch, warmup_epochs))

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()

            batch_size = batch["tes_label"].shape[0]
            for name in meter_names:
                meters[name].update(stats[name].item(), batch_size)
            collect_predictions(storage, outputs, batch)

    metrics = compute_metrics(storage)
    metrics.update({name: meter.avg for name, meter in meters.items()})
    return metrics, storage


def log_metrics(writer, split, metrics, epoch):
    if writer is None:
        return
    for key, value in metrics.items():
        writer.add_scalar("{}/{}".format(split, key), value, epoch)


def save_checkpoint(path, model, args, metrics, tes_score_max, pcs_score_max):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "metrics": metrics,
            "tes_score_max": tes_score_max,
            "pcs_score_max": pcs_score_max,
        },
        path,
    )


def train(model, criterion, train_loader, test_loader, optimizer, scheduler, device, args, writer=None):
    best_mean = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    best_metrics = None

    for epoch in range(args.epoch):
        train_metrics, _ = run_epoch(
            model,
            criterion,
            train_loader,
            device=device,
            optimizer=optimizer,
            epoch=epoch,
            warmup_epochs=args.warmup_epochs,
        )
        test_metrics, test_storage = run_epoch(
            model,
            criterion,
            test_loader,
            device=device,
            optimizer=None,
            epoch=epoch,
            warmup_epochs=args.warmup_epochs,
        )

        if scheduler is not None:
            scheduler.step()

        log_metrics(writer, "train", train_metrics, epoch)
        log_metrics(writer, "test", test_metrics, epoch)

        print(
            "Epoch {:03d} | Train mean {:.4f} (TES {:.4f}, PCS {:.4f}) | "
            "Test mean {:.4f} (TES {:.4f}, PCS {:.4f}) | loss {:.4f}".format(
                epoch,
                train_metrics["mean_srcc"],
                train_metrics["tes_srcc"],
                train_metrics["pcs_srcc"],
                test_metrics["mean_srcc"],
                test_metrics["tes_srcc"],
                test_metrics["pcs_srcc"],
                test_metrics["loss"],
            )
        )

        if test_metrics["mean_srcc"] > best_mean:
            best_mean = test_metrics["mean_srcc"]
            best_epoch = epoch
            best_metrics = test_metrics
            epochs_without_improvement = 0
            if args.ckpt_path:
                save_checkpoint(
                    args.ckpt_path,
                    model,
                    args,
                    test_metrics,
                    train_loader.dataset.tes_score_max,
                    train_loader.dataset.pcs_score_max,
                )
            if args.error_report_path:
                save_error_report(args.error_report_path, test_storage)
        else:
            epochs_without_improvement += 1

        if args.early_stop_patience > 0 and epoch >= args.warmup_epochs and epochs_without_improvement >= args.early_stop_patience:
            print("Early stopping at epoch {}.".format(epoch))
            break

    return {"best_epoch": best_epoch, "best_metrics": best_metrics}


def evaluate(model, criterion, dataloader, device, args, writer=None):
    metrics, storage = run_epoch(
        model,
        criterion,
        dataloader,
        device=device,
        optimizer=None,
        epoch=max(args.epoch - 1, 0),
        warmup_epochs=args.warmup_epochs,
    )
    log_metrics(writer, "eval", metrics, 0)
    if args.error_report_path:
        save_error_report(args.error_report_path, storage)
    return metrics


def save_error_report(path, storage):
    lines = []
    for video_id, tes_pred, tes_label, pcs_pred, pcs_label in zip(
        storage["video_id"],
        storage["tes_pred"],
        storage["tes_label"],
        storage["pcs_pred"],
        storage["pcs_label"],
    ):
        tes_delta = tes_pred - tes_label
        pcs_delta = pcs_pred - pcs_label
        if tes_delta >= 0 and pcs_delta < 0:
            case = "TES_high_PCS_low"
        elif tes_delta < 0 and pcs_delta >= 0:
            case = "TES_low_PCS_high"
        elif tes_delta >= 0 and pcs_delta >= 0:
            case = "both_high"
        else:
            case = "both_low"
        lines.append(
            "{}\t{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format(
                video_id, case, tes_pred, tes_label, pcs_pred, pcs_label
            )
        )
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
