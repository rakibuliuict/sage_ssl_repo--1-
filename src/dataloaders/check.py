# src/dataloaders/check.py
import os, sys
import torch

# Add the repo root (…\sage_ssl_repo (1)) to sys.path so "src.*" imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.dataloaders.semisup_loader import prepare_semi_supervised  # absolute import


def _shape_of(x):
    if isinstance(x, torch.Tensor):
        return tuple(x.shape)
    return None

def _summarize_sample(sample):
    """
    Return a compact dict of shapes from a single dataset item.
    Handles dicts like:
      - {'img': Tensor, 'seg': Tensor, 'patient_id': ...}
      - {'weak': {...}, 'strong': {...}, 'patient_id': ...}  (DualView)
    """
    out = {}

    if isinstance(sample, dict):
        # Dual-view unlabeled
        if "weak" in sample and "strong" in sample:
            w, s = sample["weak"], sample["strong"]
            out["weak"] = {}
            out["strong"] = {}
            for k in ("img", "t2w", "adc", "hbv", "seg"):
                if isinstance(w, dict) and k in w:
                    out["weak"][k] = _shape_of(w[k])
                if isinstance(s, dict) and k in s:
                    out["strong"][k] = _shape_of(s[k])
        else:
            # Single-view labeled/val
            for k, v in sample.items():
                if k in ("img", "t2w", "adc", "hbv", "seg"):
                    out[k] = _shape_of(v)
    return out


def main():
    in_dir = r"K:\My Drive\Code\sage_ssl_repo (1)\data\123"

    labeled_loader, unlabeled_loader, val_loader = prepare_semi_supervised(
        in_dir=in_dir,
        cache=False,
        num_workers=4,
        batch_size_labeled=1,
        batch_size_unlabeled=2,
        batch_size_val=1,
        seed=0,
    )

    labeled_ds   = labeled_loader.dataset
    unlabeled_ds = unlabeled_loader.dataset
    val_ds       = val_loader.dataset

    print("[Dataset sizes]")
    print(f"  Labeled   : {len(labeled_ds)}")
    print(f"  Unlabeled : {len(unlabeled_ds)}")
    print(f"  Val       : {len(val_ds)}")

    print("\n[Data shapes (one sample per split)]")

    # Labeled
    try:
        s = _summarize_sample(labeled_ds[0])
        print(f"  Labeled   : {s}")
    except Exception as e:
        print(f"  Labeled   : <could not infer> ({e})")

    # Unlabeled (dual-view)
    try:
        s = _summarize_sample(unlabeled_ds[0])
        print(f"  Unlabeled : {s}")
    except Exception as e:
        print(f"  Unlabeled : <could not infer> ({e})")

    # Val
    try:
        s = _summarize_sample(val_ds[0])
        print(f"  Val       : {s}")
    except Exception as e:
        print(f"  Val       : <could not infer> ({e})")


if __name__ == "__main__":
    main()




# # src/dataloaders/check.py
# import os, sys
# import torch

# # Add the repo root (…\sage_ssl_repo (1)) to sys.path so "src.*" imports work
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# from src.dataloaders.semisup_loader import prepare_semi_supervised  # absolute import


# def describe_batch(batch):
#     """Helper to describe shape and type of a batch (supports dicts, tuples, or tensors)."""
#     if isinstance(batch, torch.Tensor):
#         return f"Tensor: shape={tuple(batch.shape)}, dtype={batch.dtype}"
#     elif isinstance(batch, (list, tuple)):
#         return [describe_batch(b) for b in batch]
#     elif isinstance(batch, dict):
#         return {k: describe_batch(v) for k, v in batch.items()}
#     else:
#         return f"{type(batch).__name__}"


# def main():
#     in_dir = r"K:\My Drive\Code\sage_ssl_repo (1)\data\123"

#     labeled_loader, unlabeled_loader, val_loader = prepare_semi_supervised(
#         in_dir=in_dir,
#         cache=False,
#         num_workers=4,
#         batch_size_labeled=1,
#         batch_size_unlabeled=2,
#         batch_size_val=1,
#         seed=0,
#     )

#     print("[OK] Dataset sizes:")
#     print(f"  Labeled:   {len(labeled_loader.dataset)} samples")
#     print(f"  Unlabeled: {len(unlabeled_loader.dataset)} samples")
#     print(f"  Val:       {len(val_loader.dataset)} samples")

#     # Peek at one batch from each loader
#     print("\n[Sample batch info]")

#     for name, loader in [("Labeled", labeled_loader),
#                          ("Unlabeled", unlabeled_loader),
#                          ("Val", val_loader)]:
#         try:
#             batch = next(iter(loader))
#             desc = describe_batch(batch)
#             print(f"  {name} batch -> {desc}")
#         except Exception as e:
#             print(f"  {name} batch -> ERROR: {e}")


# if __name__ == "__main__":
#     main()
