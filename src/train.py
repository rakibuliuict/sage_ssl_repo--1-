
# # src/train.py
# import os
# import re
# import argparse
# import yaml
# import torch
# from torch.optim import AdamW, SGD
# from torch.cuda.amp import GradScaler, autocast
# from tqdm import tqdm
# import torch.nn.functional as F
# import pandas as pd
# import importlib

# from src.dataloaders.semisup_loader import prepare_semi_supervised
# from src.models.sage_ssl import SAGESSL
# from src.losses.dice_ce import DiceCELoss
# from src.losses.cdcr import CDCRLoss
# from src.utils.ema import EMA
# from src.utils.metrics import dice_score
# from src.utils.smc import self_confidence, mutual_agreement, blended_pseudolabel


# def get_args():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--data_root", type=str, required=True)
#     ap.add_argument("--config", type=str, default="configs/config.yaml")
#     ap.add_argument("--outdir", type=str, default="outputs/exp1")
#     ap.add_argument("--epochs", type=int, default=None)
#     ap.add_argument("--seed", type=int, default=None)
#     ap.add_argument("--eval_only", action="store_true")
#     ap.add_argument('--model', type=str, default='sdcl_3d', help='Model file name under src/models/')


#     # Checkpoint controls
#     # 1) --resume_ckpt: continue training with model + optimizer state from the exact checkpoint epoch
#     # 2) --pretrain_ckpt: initialize model weights ONLY, and start from a chosen epoch (e.g., 5 -> loop prints "Epoch 6")
#     ap.add_argument("--resume_ckpt", type=str, default=None, help="Path to checkpoint to fully resume from.")
#     ap.add_argument("--pretrain_ckpt", type=str, default=None, help="Path to checkpoint to load weights from (model only).")
#     ap.add_argument(
#         "--start_from_epoch",
#         type=int,
#         default=None,
#         help="Zero-based epoch index to START the loop from (e.g., 5 means next printed epoch is 6). "
#              "If omitted, will try to infer from pretrain filename like best_003.pt -> start_from_epoch=5 if you say so."
#     )
#     return ap.parse_args()


# @torch.no_grad()
# def evaluate(model, val_loader, device):
#     model.eval()
#     total, n = 0.0, 0
#     for batch in val_loader:
#         x = torch.cat([batch["t2w"], batch["adc"], batch["hbv"]], dim=1).to(device, non_blocking=True)
#         y = batch["seg"].squeeze(1).long().to(device, non_blocking=True)

#         out = model(xL=x, xU=None, train=False)
#         logits = out["pL"]
#         num_classes = logits.shape[1]
#         y_one = F.one_hot(y, num_classes=num_classes)
#         if logits.ndim == 4:                         # [B,C,H,W]
#             y_one = y_one.permute(0, 3, 1, 2).float()
#         else:                                        # [B,C,D,H,W]
#             y_one = y_one.permute(0, 4, 1, 2, 3).float()

#         total += dice_score(logits, y_one)
#         n += 1
#     return total / max(n, 1)

# def _save_dice_to_excel(outdir: str, epoch_num: int, dice_value: float, fname: str = "metrics.xlsx"):
#     """
#     Append/overwrite a row (epoch, dice) in an Excel file located at `outdir/fname`.
#     If the file exists, we deduplicate by epoch (keeping the latest value).
#     """
#     os.makedirs(outdir, exist_ok=True)
#     path = os.path.join(outdir, fname)
#     row = pd.DataFrame([{"epoch": int(epoch_num), "dice": float(dice_value)}])

#     if os.path.exists(path):
#         try:
#             old = pd.read_excel(path)
#             df = pd.concat([old, row], ignore_index=True)
#         except Exception:
#             # If the existing file is unreadable, start fresh
#             df = row
#     else:
#         df = row

#     # keep last occurrence for each epoch
#     df = df.drop_duplicates(subset=["epoch"], keep="last").sort_values("epoch")
#     # write (overwrite file) to keep a clean single sheet
#     with pd.ExcelWriter(path, engine="openpyxl", mode="w") as w:
#         df.to_excel(w, index=False, sheet_name="metrics")


# def _extract_views(batch_u):
#     if batch_u is None:
#         return None, None
#     if "view1" in batch_u and "view2" in batch_u:
#         return batch_u["view1"], batch_u["view2"]
#     if "weak" in batch_u and "strong" in batch_u:
#         return batch_u["weak"], batch_u["strong"]
#     return None, None


# def _infer_epoch_from_name(path):
#     """
#     Try to infer a 1-based epoch number from filenames like:
#       best_003.pt, epoch_010.pt
#     Returns an int (1-based) or None if not found.
#     """
#     m = re.search(r"(?:best|epoch)_(\d+)\.pt$", os.path.basename(path))
#     if not m:
#         return None
#     return int(m.group(1))


# def main():
#     args = get_args()
#     with open(args.config, "r") as f:
#         cfg = yaml.safe_load(f)

#     # CLI overrides
#     if args.epochs is not None:
#         cfg["train"]["epochs"] = args.epochs
#     if args.seed is not None:
#         cfg["seed"] = args.seed

#     os.makedirs(args.outdir, exist_ok=True)
#     ckpt_dir = os.path.join(args.outdir, cfg["io"]["ckpt_dir"])
#     os.makedirs(ckpt_dir, exist_ok=True)

#     torch.manual_seed(cfg["seed"])
#     torch.cuda.manual_seed_all(cfg["seed"])
#     torch.backends.cudnn.benchmark = True

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # --- Dataloaders ---
#     labeled_loader, unlabeled_loader, val_loader = prepare_semi_supervised(
#         in_dir=args.data_root,
#         cache=cfg["train"]["cache"],
#         cache_rate=cfg["train"]["cache_rate"],
#         num_workers=cfg["io"].get("num_workers", 4),
#         batch_size_labeled=cfg["train"]["batch_labeled"],
#         batch_size_unlabeled=cfg["train"]["batch_unlabeled"],
#         batch_size_val=cfg["train"]["batch_val"],
#         seed=cfg["seed"],
#     )

#     use_unsup = True
#     try:
#         use_unsup = len(unlabeled_loader.dataset) > 0
#     except Exception:
#         pass

#     # --- Model / Opt / Losses ---
#     model = SAGESSL(
#         num_classes=cfg["num_classes"],
#         in_ch=cfg["in_modalities"],
#         chs=cfg["model"]["channels"],
#         rank=cfg["model"]["rank"],
#     ).to(device)

#     if cfg["optim"]["name"].lower() == "adamw":
#         optim = AdamW(model.parameters(), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])
#     else:
#         optim = SGD(model.parameters(), lr=cfg["optim"]["lr"], momentum=0.9, weight_decay=cfg["optim"]["weight_decay"])

#     scaler = GradScaler(enabled=cfg["train"]["amp"])
#     ema = EMA(model, decay=cfg["optim"]["ema"])

#     sup_loss = DiceCELoss()
#     cdcr = CDCRLoss()

#     # --- Checkpoint logic ---
#     start_epoch, best_val = 0, -1.0

#     # 1) Full resume (model + optimizer) from an exact training checkpoint
#     if args.resume_ckpt:
#         if not os.path.isfile(args.resume_ckpt):
#             raise FileNotFoundError(f"--resume_ckpt not found: {args.resume_ckpt}")
#         ckpt = torch.load(args.resume_ckpt, map_location="cpu")
#         model.load_state_dict(ckpt["model"], strict=True)
#         if "optim" in ckpt and ckpt["optim"] is not None:
#             optim.load_state_dict(ckpt["optim"])
#         start_epoch = ckpt.get("epoch", 0)
#         best_val = ckpt.get("best_val", -1.0)
#         print(f"[Resume] {args.resume_ckpt} -> start_epoch={start_epoch}, best_val={best_val:.4f}")

#     # 2) Pretrain initialization: model weights ONLY, fresh optimizer; choose where to start
#     elif args.pretrain_ckpt:
#         if not os.path.isfile(args.pretrain_ckpt):
#             raise FileNotFoundError(f"--pretrain_ckpt not found: {args.pretrain_ckpt}")
#         ckpt = torch.load(args.pretrain_ckpt, map_location="cpu")
#         # Support both raw state_dict or dict with "model"
#         state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
#         missing, unexpected = model.load_state_dict(state, strict=False)
#         if missing:
#             print(f"[Pretrain] Missing keys: {len(missing)} (showing up to 10): {missing[:10]}")
#         if unexpected:
#             print(f"[Pretrain] Unexpected keys: {len(unexpected)} (showing up to 10): {unexpected[:10]}")

#         # Choose start epoch:
#         # If user passed --start_from_epoch, use that (zero-based).
#         # Otherwise, infer from filename if possible, else 0.
#         if args.start_from_epoch is not None:
#             start_epoch = args.start_from_epoch
#         else:
#             inferred_1_based = _infer_epoch_from_name(args.pretrain_ckpt)
#             # Common workflow (your example):
#             # trained to epoch 5, want to START FROM epoch 5 (zero-based), so the loop prints "Epoch 6".
#             # Pass --start_from_epoch 5 explicitly to avoid ambiguity.
#             start_epoch = 0 if inferred_1_based is None else max(inferred_1_based - 1, 0)

#         best_val = -1.0  # new best tracking when fine-tuning
#         print(f"[Pretrain] Loaded weights from {args.pretrain_ckpt} -> start_epoch={start_epoch}")

#     # Eval-only branch works with whichever path you used above (or randomly initialized)
#     if args.eval_only:
#         val_dice = evaluate(model, val_loader, device)
#         print(f"[Eval] Dice: {val_dice:.4f}")
#         print(f"[Eval] Dice: {val_dice:.4f}")
#         _save_dice_to_excel(args.outdir, -1, float(val_dice))
#         return

#     # --- Train ---
#     for epoch in range(start_epoch, cfg["train"]["epochs"]):
#         model.train()
#         it_un = iter(unlabeled_loader) if use_unsup else None

#         pbar = tqdm(labeled_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}")
#         for batch_l in pbar:
#             # Labeled
#             xL = torch.cat([batch_l["t2w"], batch_l["adc"], batch_l["hbv"]], dim=1).to(device, non_blocking=True)
#             yL = batch_l["seg"].squeeze(1).long().to(device, non_blocking=True)

#             # Unlabeled dual views (optional)
#             xUw = xUs = None
#             if use_unsup:
#                 try:
#                     batch_u = next(it_un)
#                 except StopIteration:
#                     it_un = iter(unlabeled_loader)
#                     batch_u = next(it_un)
#                 v1, v2 = _extract_views(batch_u)
#                 if v1 is not None and v2 is not None:
#                     xUw = torch.cat([v1["t2w"], v1["adc"], v1["hbv"]], dim=1).to(device, non_blocking=True)
#                     xUs = torch.cat([v2["t2w"], v2["adc"], v2["hbv"]], dim=1).to(device, non_blocking=True)

#             optim.zero_grad(set_to_none=True)
#             with autocast(enabled=cfg["train"]["amp"]):
#                 out_w = model(xL=xL, xU=xUw, train=True)
#                 L_sup = sup_loss(out_w["pL"], yL)

#                 L_unsup = torch.tensor(0.0, device=device)
#                 L_cdcr = torch.tensor(0.0, device=device)
#                 L_ab   = torch.tensor(0.0, device=device)
#                 L_nc   = torch.tensor(0.0, device=device)

#                 if use_unsup and xUs is not None:
#                     out_s = model(xL=None, xU=xUs, train=True)

#                     alpha_self = (
#                         self_confidence(out_w["pU"]).clamp(0, 1)
#                         + self_confidence(out_s["pU"]).clamp(0, 1)
#                     ) / 2.0
#                     alpha_mut = mutual_agreement(out_w["pU"], out_s["pU"]).float()
#                     alpha = 0.5 * alpha_self + 0.5 * alpha_mut

#                     pseudo, mask = blended_pseudolabel(
#                         out_w["pU"], out_s["pU"], alpha, conf_drop_fraction=cfg["smc"]["conf_drop_fraction"]
#                     )
#                     # If your loss supports masks, apply them with 'mask' here as needed.
#                     L_unsup = sup_loss(out_w["pU"], pseudo)

#                     L_cdcr = cdcr(out_w["pU"], out_w.get("pU_noisy", out_w["pU"]))
#                     gate = model.ab_head(out_w["ab_feats"]).detach()
#                     L_ab = 0.5 * gate.mean()
#                     L_nc = torch.mean(
#                         (torch.softmax(out_w["pU"], 1) - torch.softmax(out_s["pU"], 1)).abs()
#                     )

#                 loss = (
#                     L_sup
#                     + cfg["loss_weights"]["unsup"] * L_unsup
#                     + cfg["loss_weights"]["cdcr"] * L_cdcr
#                     + cfg["loss_weights"]["ab"] * L_ab
#                     + cfg["loss_weights"]["nc"] * L_nc
#                 )

#             scaler.scale(loss).backward()
#             scaler.step(optim)
#             scaler.update()
#             ema.update(model)

#             pbar.set_postfix(loss=float(loss.detach().cpu()))

#         # --- EMA validation ---
#         ema.apply_to(model)
#         val_dice = evaluate(model, val_loader, device)

#         # save (epoch is 1-based in your prints)
#         _save_dice_to_excel(args.outdir, epoch + 1, float(val_dice))

#         # --- Save top-2 best checkpoints ---
#         if not hasattr(main, "best_models"):
#             main.best_models = []

#         if val_dice > best_val:
#             best_val = val_dice
#             model_path = os.path.join(ckpt_dir, f"best_{epoch+1:03d}.pt")
#             torch.save(
#                 {
#                     "model": model.state_dict(),
#                     "optim": optim.state_dict(),
#                     "epoch": epoch + 1,   # next start index if resuming
#                     "best_val": best_val,
#                 },
#                 model_path,
#             )
#             main.best_models.append(model_path)
#             print(f"[Checkpoint] Saved new best model: {model_path}")

#             if len(main.best_models) > 2:
#                 oldest = main.best_models.pop(0)
#                 try:
#                     os.remove(oldest)
#                     print(f"[Cleanup] Removed older best model: {oldest}")
#                 except FileNotFoundError:
#                     pass

#         print(f"[Val] Epoch {epoch+1}: Dice={val_dice:.4f} (best={best_val:.4f})")


# if __name__ == "__main__":
#     main()


# src/train.py
import os
import re
import argparse
import yaml
import torch
from torch.optim import AdamW, SGD
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
import importlib
import inspect

from src.dataloaders.semisup_loader import prepare_semi_supervised
from src.utils.metrics import dice_score

# ---- Optional SAGE bits (only used when model is not SDCL) ----
from src.losses.dice_ce import DiceCELoss
from src.losses.cdcr import CDCRLoss
from src.utils.ema import EMA
from src.utils.smc import self_confidence, mutual_agreement, blended_pseudolabel


# =========================
# Args
# =========================
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    ap.add_argument("--outdir", type=str, default="outputs/exp1")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--eval_only", action="store_true")

    # NEW: choose model by file name under src/models/
    # examples: --model sdcl_3d   or   --model sage_ssl
    ap.add_argument("--model", type=str, default="sdcl_3d",
                    help="Model file name in src/models (e.g., sdcl_3d, sage_ssl, sage_ssl_3d)")

    # Checkpoint controls
    ap.add_argument("--resume_ckpt", type=str, default=None, help="Resume training from checkpoint (model+optim).")
    ap.add_argument("--pretrain_ckpt", type=str, default=None, help="Initialize model weights only from checkpoint.")
    ap.add_argument("--start_from_epoch", type=int, default=None,
                    help="Zero-based starting epoch (overrides automatic inference).")
    return ap.parse_args()


# =========================
# Utilities
# =========================
def _extract_views(batch_u):
    if batch_u is None:
        return None, None
    if "view1" in batch_u and "view2" in batch_u:
        return batch_u["view1"], batch_u["view2"]
    if "weak" in batch_u and "strong" in batch_u:
        return batch_u["weak"], batch_u["strong"]
    return None, None


def _infer_epoch_from_name(path):
    m = re.search(r"(?:best|epoch)_(\d+)\.pt$", os.path.basename(path))
    if not m:
        return None
    return int(m.group(1))


def _save_dice_to_excel(outdir: str, epoch_num: int, dice_value: float, fname: str = "metrics.xlsx"):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, fname)
    row = pd.DataFrame([{"epoch": int(epoch_num), "dice": float(dice_value)}])

    if os.path.exists(path):
        try:
            old = pd.read_excel(path)
            df = pd.concat([old, row], ignore_index=True)
        except Exception:
            df = row
    else:
        df = row

    df = df.drop_duplicates(subset=["epoch"], keep="last").sort_values("epoch")
    with pd.ExcelWriter(path, engine="openpyxl", mode="w") as w:
        df.to_excel(w, index=False, sheet_name="metrics")


def _load_model_by_name(name: str, num_classes: int, in_ch: int, channels):
    """
    Import src.models.<name> and instantiate the most likely class.
    Heuristics:
      - preferred class names: SDCL3D / SAGESSL3D / SAGESSL
      - otherwise, first nn.Module exported.
    """
    module = importlib.import_module(f"src.models.{name}")

    preferred = ["SDCL3D", "SAGESSL3D", "SAGESSL"]
    for cls_name in preferred:
        if hasattr(module, cls_name):
            cls = getattr(module, cls_name)
            return cls(num_classes=num_classes, in_ch=in_ch, chs=tuple(channels))

    # fallback: pick the first nn.Module subclass in module globals
    candidates = []
    for k, v in module.__dict__.items():
        if inspect.isclass(v):
            try:
                import torch.nn as nn
                if issubclass(v, nn.Module):
                    candidates.append(v)
            except Exception:
                pass
    if not candidates:
        raise RuntimeError(f"No nn.Module class found in src.models.{name}")

    cls = candidates[0]
    # best effort: accept (num_classes, in_ch, chs=...)
    try:
        return cls(num_classes=num_classes, in_ch=in_ch, chs=tuple(channels))
    except TypeError:
        return cls(num_classes=num_classes, in_ch=in_ch)


def _is_sdcl(name: str) -> bool:
    return "sdcl" in name.lower()


# =========================
# Evaluation
# =========================
@torch.no_grad()
def evaluate_generic(model, val_loader, device, is_sdcl: bool):
    model.eval()
    total, n = 0.0, 0
    for batch in val_loader:
        x = torch.cat([batch["t2w"], batch["adc"], batch["hbv"]], dim=1).to(device, non_blocking=True)
        y = batch["seg"].squeeze(1).long().to(device, non_blocking=True)

        if is_sdcl:
            s1, s2 = model(x)
            logits = (s1 + s2) / 2.0
        else:
            out = model(xL=x, xU=None, train=False)
            logits = out["pL"]

        num_classes = logits.shape[1]
        y_one = F.one_hot(y, num_classes=num_classes)
        if logits.ndim == 4:  # [B,C,H,W]
            y_one = y_one.permute(0, 3, 1, 2).float()
        else:                 # [B,C,H,W,D]
            y_one = y_one.permute(0, 4, 1, 2, 3).float()

        total += dice_score(logits, y_one)
        n += 1
    return total / max(n, 1)


# =========================
# Main
# =========================
def main():
    args = get_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.seed is not None:
        cfg["seed"] = args.seed

    os.makedirs(args.outdir, exist_ok=True)
    ckpt_dir = os.path.join(args.outdir, cfg["io"]["ckpt_dir"])
    os.makedirs(ckpt_dir, exist_ok=True)

    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Dataloaders (unchanged) ---
    labeled_loader, unlabeled_loader, val_loader = prepare_semi_supervised(
        in_dir=args.data_root,
        cache=cfg["train"]["cache"],
        cache_rate=cfg["train"]["cache_rate"],
        num_workers=cfg["io"].get("num_workers", 4),
        batch_size_labeled=cfg["train"]["batch_labeled"],
        batch_size_unlabeled=cfg["train"]["batch_unlabeled"],
        batch_size_val=cfg["train"]["batch_val"],
        seed=cfg["seed"],
    )
    use_unsup = True
    try:
        use_unsup = len(unlabeled_loader.dataset) > 0
    except Exception:
        pass

    # --- Model ---
    model = _load_model_by_name(
        args.model,
        num_classes=cfg["num_classes"],
        in_ch=cfg["in_modalities"],
        channels=cfg["model"]["channels"],
    ).to(device)
    is_sdcl = _is_sdcl(args.model)

    # --- Optimizer ---
    if cfg["optim"]["name"].lower() == "adamw":
        optim = AdamW(model.parameters(), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])
    else:
        optim = SGD(model.parameters(), lr=cfg["optim"]["lr"], momentum=0.9, weight_decay=cfg["optim"]["weight_decay"])

    scaler = GradScaler(enabled=cfg["train"]["amp"])

    # --- Losses ---
    if is_sdcl:
        # SDCL path
        from src.losses.sdcl import dice_ce_3d, SDCLUnlabeledLoss
        sup_loss_fn = dice_ce_3d
        sdcl_loss = SDCLUnlabeledLoss(
            w_agree=cfg["loss_weights"].get("agree", 1.0),
            w_disagree=cfg["loss_weights"].get("disagree", 0.5),
            temp=cfg.get("sdcl", {}).get("temp", 1.0),
        )
        ema = None  # not used for SDCL baseline
    else:
        # Legacy SAGE path
        sup_loss = DiceCELoss()
        cdcr = CDCRLoss()
        ema = EMA(model, decay=cfg["optim"]["ema"])

    # --- Checkpoint logic ---
    start_epoch, best_val = 0, -1.0

    if args.resume_ckpt:
        if not os.path.isfile(args.resume_ckpt):
            raise FileNotFoundError(f"--resume_ckpt not found: {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        if "optim" in ckpt and ckpt["optim"] is not None:
            optim.load_state_dict(ckpt["optim"])
        start_epoch = ckpt.get("epoch", 0)
        best_val = ckpt.get("best_val", ckpt.get("best", -1.0))
        print(f"[Resume] {args.resume_ckpt} -> start_epoch={start_epoch}, best={best_val:.4f}")

    elif args.pretrain_ckpt:
        if not os.path.isfile(args.pretrain_ckpt):
            raise FileNotFoundError(f"--pretrain_ckpt not found: {args.pretrain_ckpt}")
        ckpt = torch.load(args.pretrain_ckpt, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[Pretrain] Missing keys: {len(missing)} (showing up to 10): {missing[:10]}")
        if unexpected:
            print(f"[Pretrain] Unexpected keys: {len(unexpected)} (showing up to 10): {unexpected[:10]}")
        if args.start_from_epoch is not None:
            start_epoch = args.start_from_epoch
        else:
            inferred = _infer_epoch_from_name(args.pretrain_ckpt)
            start_epoch = 0 if inferred is None else max(inferred - 1, 0)
        best_val = -1.0
        print(f"[Pretrain] Loaded weights from {args.pretrain_ckpt} -> start_epoch={start_epoch}")

    # --- Eval only ---
    if args.eval_only:
        val_dice = evaluate_generic(model, val_loader, device, is_sdcl=is_sdcl)
        print(f"[Eval] Dice: {val_dice:.4f}")
        _save_dice_to_excel(args.outdir, -1, float(val_dice))
        return

    # =========================
    # Train
    # =========================
    def ramp(epoch, max_w, warm=30):
        return max_w * min(1.0, (epoch + 1) / float(warm))

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        model.train()
        it_un = iter(unlabeled_loader) if use_unsup else None
        pbar = tqdm(labeled_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}")

        for b in pbar:
            # Labeled
            xL = torch.cat([b["t2w"], b["adc"], b["hbv"]], dim=1).to(device, non_blocking=True)
            yL = b["seg"].squeeze(1).long().to(device, non_blocking=True)

            # Unlabeled (optional)
            xUw = xUs = None
            if use_unsup:
                try:
                    bu = next(it_un)
                except StopIteration:
                    it_un = iter(unlabeled_loader)
                    bu = next(it_un)
                v1, v2 = _extract_views(bu)
                if v1 is not None and v2 is not None:
                    xUw = torch.cat([v1["t2w"], v1["adc"], v1["hbv"]], dim=1).to(device, non_blocking=True)
                    xUs = torch.cat([v2["t2w"], v2["adc"], v2["hbv"]], dim=1).to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with autocast(enabled=cfg["train"]["amp"]):
                if is_sdcl:
                    # ---- SDCL path ----
                    # Supervised on student-1 logits
                    s1L, s2L = model(xL)
                    L_sup = sup_loss_fn(s1L, yL)

                    L_unsup = torch.tensor(0.0, device=device)
                    if use_unsup and (xUw is not None and xUs is not None):
                        s1w, s2w = model(xUw)
                        s1s, s2s = model(xUs)
                        L_u_w = sdcl_loss(s1w, s2w)
                        L_u_s = sdcl_loss(s1s, s2s) * 0.5  # optional lower weight on strong view
                        lam_unsup = ramp(epoch, cfg["loss_weights"].get("unsup", 2.0),
                                         warm=cfg["train"].get("unsup_warm", 30))
                        L_unsup = lam_unsup * (L_u_w + L_u_s)

                    loss = L_sup + L_unsup

                else:
                    # ---- Legacy SAGE path ----
                    out_w = model(xL=xL, xU=xUw, train=True)
                    L_sup = DiceCELoss()(out_w["pL"], yL)

                    L_unsup = torch.tensor(0.0, device=device)
                    L_cdcr = torch.tensor(0.0, device=device)
                    L_ab   = torch.tensor(0.0, device=device)
                    L_nc   = torch.tensor(0.0, device=device)

                    if use_unsup and xUs is not None:
                        out_s = model(xL=None, xU=xUs, train=True)

                        alpha_self = (self_confidence(out_w["pU"]).clamp(0,1)
                                      + self_confidence(out_s["pU"]).clamp(0,1)) / 2.0
                        alpha_mut  = mutual_agreement(out_w["pU"], out_s["pU"]).float()
                        alpha = 0.5*alpha_self + 0.5*alpha_mut

                        pseudo, mask = blended_pseudolabel(
                            out_w["pU"], out_s["pU"], alpha,
                            conf_drop_fraction=cfg["smc"]["conf_drop_fraction"])
                        L_unsup = DiceCELoss()(out_w["pU"], pseudo)

                        L_cdcr = CDCRLoss()(out_w["pU"], out_w.get("pU_noisy", out_w["pU"]))
                        gate = model.ab_head(out_w["ab_feats"]).detach()
                        L_ab = 0.5 * gate.mean()
                        L_nc = torch.mean((torch.softmax(out_w["pU"],1) - torch.softmax(out_s["pU"],1)).abs())

                    loss = (L_sup
                            + cfg["loss_weights"]["unsup"] * L_unsup
                            + cfg["loss_weights"]["cdcr"] * L_cdcr
                            + cfg["loss_weights"]["ab"]   * L_ab
                            + cfg["loss_weights"]["nc"]   * L_nc)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()

            # EMA is only meaningful for SAGE branch here
            if not is_sdcl:
                EMA(model, decay=cfg["optim"]["ema"]).update(model)

            pbar.set_postfix(loss=float(loss.detach().cpu()))

        # ----- Validation -----
        val_dice = evaluate_generic(model, val_loader, device, is_sdcl=is_sdcl)

        # Save metrics
        _save_dice_to_excel(args.outdir, epoch + 1, float(val_dice))

        # Keep top-2 best
        if not hasattr(main, "best_models"):
            main.best_models = []
        best_key = "best_val"  # also compatible with "best" from older checkpoints

        if len(main.best_models) < 2:
            best_so_far = max([m[1] for m in main.best_models], default=-1.0)
        else:
            best_so_far = max([m[1] for m in main.best_models])

        if val_dice > best_so_far:
            ckpt_path = os.path.join(ckpt_dir, f"best_{epoch+1:03d}.pt")
            torch.save({
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "epoch": epoch + 1,
                best_key: float(val_dice),
            }, ckpt_path)
            main.best_models.append((ckpt_path, float(val_dice)))
            main.best_models = sorted(main.best_models, key=lambda x: -x[1])[:2]
            # cleanup extras (keep top-2)
            for p, _ in main.best_models[2:]:
                try: os.remove(p)
                except FileNotFoundError: pass

        print(f"[Val] Epoch {epoch+1}: Dice={val_dice:.4f}")

    # end for epochs


if __name__ == "__main__":
    main()
