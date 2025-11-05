# src/train.py
import os
import argparse
import yaml
import torch
from torch.optim import AdamW, SGD
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.dataloaders.semisup_loader import prepare_semi_supervised
from src.models.sage_ssl import SAGESSL
from src.losses.dice_ce import DiceCELoss
from src.losses.cdcr import CDCRLoss
from src.utils.ema import EMA
from src.utils.metrics import dice_score
from src.utils.smc import self_confidence, mutual_agreement, blended_pseudolabel


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    ap.add_argument("--outdir", type=str, default="outputs/exp1")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--ckpt", type=str, default=None)
    return ap.parse_args()


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    total, n = 0.0, 0
    for batch in val_loader:
        x = torch.cat([batch["t2w"], batch["adc"], batch["hbv"]], dim=1).to(device, non_blocking=True)
        y = batch["seg"].squeeze(1).long().to(device, non_blocking=True)
        out = model(xL=x, xU=None, train=False)
        y_one = torch.nn.functional.one_hot(y, num_classes=out["pL"].shape[1]).permute(0, 3, 1, 2).float()
        total += dice_score(out["pL"], y_one)
        n += 1
    return total / max(n, 1)


def _extract_views(batch_u):
    """
    Support either {'view1','view2'} or {'weak','strong'} keys from the unlabeled loader.
    Returns (view_weak_dict, view_strong_dict) or (None, None) if not available.
    """
    if batch_u is None:
        return None, None
    if "view1" in batch_u and "view2" in batch_u:
        return batch_u["view1"], batch_u["view2"]
    if "weak" in batch_u and "strong" in batch_u:
        return batch_u["weak"], batch_u["strong"]
    return None, None


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

    # --- Dataloaders (no labeled_fraction) ---
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

    # Train supervised-only if unlabeled is empty
    use_unsup = True
    try:
        use_unsup = len(unlabeled_loader.dataset) > 0
    except Exception:
        pass

    # --- Model / Opt / Losses ---
    model = SAGESSL(
        num_classes=cfg["num_classes"],
        in_ch=cfg["in_modalities"],
        chs=cfg["model"]["channels"],
        rank=cfg["model"]["rank"],
    ).to(device)

    if cfg["optim"]["name"].lower() == "adamw":
        optim = AdamW(model.parameters(), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])
    else:
        optim = SGD(model.parameters(), lr=cfg["optim"]["lr"], momentum=0.9, weight_decay=cfg["optim"]["weight_decay"])

    scaler = GradScaler(enabled=cfg["train"]["amp"])
    ema = EMA(model, decay=cfg["optim"]["ema"])

    sup_loss = DiceCELoss()
    cdcr = CDCRLoss()

    # --- Resume (optional) ---
    start_epoch, best_val = 0, -1.0
    if args.ckpt and os.path.isfile(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        start_epoch = ckpt.get("epoch", 0)
        best_val = ckpt.get("best_val", -1.0)
        print(f"[Resume] Loaded {args.ckpt} (epoch {start_epoch}, best {best_val:.4f})")

    if args.eval_only:
        val_dice = evaluate(model, val_loader, device)
        print(f"[Eval] Dice: {val_dice:.4f}")
        return

    # --- Train ---
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        model.train()
        it_un = iter(unlabeled_loader) if use_unsup else None

        pbar = tqdm(labeled_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}")
        for batch_l in pbar:
            # Labeled
            xL = torch.cat([batch_l["t2w"], batch_l["adc"], batch_l["hbv"]], dim=1).to(device, non_blocking=True)
            yL = batch_l["seg"].squeeze(1).long().to(device, non_blocking=True)

            # Unlabeled dual views (optional)
            xUw = xUs = None
            if use_unsup:
                try:
                    batch_u = next(it_un)
                except StopIteration:
                    it_un = iter(unlabeled_loader)
                    batch_u = next(it_un)
                v1, v2 = _extract_views(batch_u)
                if v1 is not None and v2 is not None:
                    xUw = torch.cat([v1["t2w"], v1["adc"], v1["hbv"]], dim=1).to(device, non_blocking=True)
                    xUs = torch.cat([v2["t2w"], v2["adc"], v2["hbv"]], dim=1).to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with autocast(enabled=cfg["train"]["amp"]):
                out_w = model(xL=xL, xU=xUw, train=True)
                L_sup = sup_loss(out_w["pL"], yL)

                L_unsup = torch.tensor(0.0, device=device)
                L_cdcr = torch.tensor(0.0, device=device)
                L_ab   = torch.tensor(0.0, device=device)
                L_nc   = torch.tensor(0.0, device=device)

                if use_unsup and xUs is not None:
                    out_s = model(xL=None, xU=xUs, train=True)

                    alpha_self = (
                        self_confidence(out_w["pU"]).clamp(0, 1)
                        + self_confidence(out_s["pU"]).clamp(0, 1)
                    ) / 2.0
                    alpha_mut = mutual_agreement(out_w["pU"], out_s["pU"]).float()
                    alpha = 0.5 * alpha_self + 0.5 * alpha_mut

                    pseudo, mask = blended_pseudolabel(
                        out_w["pU"], out_s["pU"], alpha, conf_drop_fraction=cfg["smc"]["conf_drop_fraction"]
                    )
                    # If your loss supports masks, apply them here.
                    L_unsup = sup_loss(out_w["pU"], pseudo)

                    L_cdcr = cdcr(out_w["pU"], out_w.get("pU_noisy", out_w["pU"]))
                    gate = model.ab_head(out_w["ab_feats"]).detach()
                    L_ab = 0.5 * gate.mean()
                    L_nc = torch.mean(
                        (torch.softmax(out_w["pU"], 1) - torch.softmax(out_s["pU"], 1)).abs()
                    )

                loss = (
                    L_sup
                    + cfg["loss_weights"]["unsup"] * L_unsup
                    + cfg["loss_weights"]["cdcr"] * L_cdcr
                    + cfg["loss_weights"]["ab"] * L_ab
                    + cfg["loss_weights"]["nc"] * L_nc
                )

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            ema.update(model)

            pbar.set_postfix(loss=float(loss.detach().cpu()))

        # --- EMA validation ---
        ema.apply_to(model)
        val_dice = evaluate(model, val_loader, device)

        # --- Save checkpoints ---
        if val_dice > best_val:
            best_val = val_dice
            torch.save(
                {"model": model.state_dict(), "optim": optim.state_dict(), "epoch": epoch + 1, "best_val": best_val},
                os.path.join(ckpt_dir, "best.pt"),
            )
        torch.save(
            {"model": model.state_dict(), "optim": optim.state_dict(), "epoch": epoch + 1, "best_val": best_val},
            os.path.join(ckpt_dir, f"epoch_{epoch+1}.pt"),
        )
        print(f"[Val] Epoch {epoch+1}: Dice={val_dice:.4f} (best={best_val:.4f})")


if __name__ == "__main__":
    main()
