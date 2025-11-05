import os
from glob import glob
from typing import List, Dict, Optional, Callable, Tuple
from monai.data import DataLoader, Dataset, CacheDataset
from torch.utils.data import Dataset as TorchDataset
from monai.utils import set_determinism

# Imports aligned with your earlier structure (no "src." prefix)
from ..training_setup.augmentations.train_augment import get_train_transforms
from ..training_setup.augmentations.test_augment import get_test_transforms
try:
    from ..training_setup.augmentations.unlabeled_augment import (
        get_unlabeled_weak_transforms, get_unlabeled_strong_transforms
    )
except Exception:
    # Fallback if file not present
    def get_unlabeled_weak_transforms(): return get_train_transforms()
    def get_unlabeled_strong_transforms(): return get_train_transforms()

# ---------- path utils ----------
def _pid_from_path(p: str) -> str:
    # "PID_xxx_*.nii.gz" -> "PID"
    return os.path.basename(p).split("_")[0]

def _index_by_pid(paths: List[str]) -> Dict[str, str]:
    return {_pid_from_path(p): p for p in paths}

def _collect_labeled(in_dir: str) -> List[Dict]:
    """
    Build labeled samples from TrainVolume/* joined by pid:
    returns [{"patient_id", "t2w","adc","hbv","seg"}, ...]
    """
    t2w = sorted(glob(os.path.join(in_dir, "TrainVolumes", "t2w", "*.nii.gz")))
    adc = sorted(glob(os.path.join(in_dir, "TrainVolumes", "adc", "*.nii.gz")))
    hbv = sorted(glob(os.path.join(in_dir, "TrainVolumes", "hbv", "*.nii.gz")))
    seg = sorted(glob(os.path.join(in_dir, "Binary_TrainSegmentation", "*.nii.gz")))

    i_t2w = _index_by_pid(t2w)
    i_adc = _index_by_pid(adc)
    i_hbv = _index_by_pid(hbv)
    i_seg = _index_by_pid(seg)

    common = sorted(set(i_t2w) & set(i_adc) & set(i_hbv) & set(i_seg))
    samples = [{
        "patient_id": pid,
        "t2w": i_t2w[pid],
        "adc": i_adc[pid],
        "hbv": i_hbv[pid],
        "seg": i_seg[pid],
    } for pid in common]
    return samples

def _collect_val(in_dir: str) -> List[Dict]:
    """
    Build validation samples from ValidVolume/* joined by pid.
    """
    t2w = sorted(glob(os.path.join(in_dir, "ValidVolumes", "t2w", "*.nii.gz")))
    adc = sorted(glob(os.path.join(in_dir, "ValidVolumes", "adc", "*.nii.gz")))
    hbv = sorted(glob(os.path.join(in_dir, "ValidVolumes", "hbv", "*.nii.gz")))
    seg = sorted(glob(os.path.join(in_dir, "Binary_ValidSegmentation", "*.nii.gz")))

    i_t2w = _index_by_pid(t2w)
    i_adc = _index_by_pid(adc)
    i_hbv = _index_by_pid(hbv)
    i_seg = _index_by_pid(seg)

    common = sorted(set(i_t2w) & set(i_adc) & set(i_hbv) & set(i_seg))
    samples = [{
        "patient_id": pid,
        "t2w": i_t2w[pid],
        "adc": i_adc[pid],
        "hbv": i_hbv[pid],
        "seg": i_seg[pid],
    } for pid in common]
    return samples

def _collect_unlabeled(in_dir: str) -> List[Dict]:
    """
    Build unlabeled samples from UnlabeledVolume/* joined by pid.
    returns [{"patient_id","t2w","adc","hbv"}, ...]
    """
    t2w = sorted(glob(os.path.join(in_dir, "UnlabeledVolumes", "t2w", "*.nii.gz")))
    adc = sorted(glob(os.path.join(in_dir, "UnlabeledVolumes", "adc", "*.nii.gz")))
    hbv = sorted(glob(os.path.join(in_dir, "UnlabeledVolumes", "hbv", "*.nii.gz")))

    i_t2w = _index_by_pid(t2w)
    i_adc = _index_by_pid(adc)
    i_hbv = _index_by_pid(hbv)

    common = sorted(set(i_t2w) & set(i_adc) & set(i_hbv))
    samples = [{
        "patient_id": pid,
        "t2w": i_t2w[pid],
        "adc": i_adc[pid],
        "hbv": i_hbv[pid],
    } for pid in common]
    return samples

# ---------- Dual-view unlabeled dataset ----------
class DualViewUnlabeledDataset(TorchDataset):
    """
    Returns a dict with:
      {
        "weak":   {"t2w": tensor, "adc": tensor, "hbv": tensor, "patient_id": str, ...},
        "strong": {"t2w": tensor, "adc": tensor, "hbv": tensor, "patient_id": str, ...},
        "patient_id": str
      }
    """
    def __init__(
        self,
        data: List[Dict],
        weak_transform: Callable,
        strong_transform: Callable,
        cache: bool = False,
        cache_rate: float = 1.0,
        num_workers: int = 0,
    ):
        self.data = data
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

        # We intentionally keep the base dataset "raw" (no transform),
        # because our weak/strong pipelines start with LoadImaged.
        base_cls = CacheDataset if cache else Dataset
        base_kwargs = dict(data=data, transform=None)
        if cache:
            base_kwargs.update(cache_rate=cache_rate, num_workers=num_workers)
        self.base_ds = base_cls(**base_kwargs)

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx: int):
        sample = self.base_ds[idx]  # dict with file paths: "t2w","adc","hbv", plus "patient_id"
        pid = sample.get("patient_id", None)

        # Apply transforms independently for weak & strong views
        weak = self.weak_transform({**sample})
        strong = self.strong_transform({**sample})

        # Attach patient_id for traceability
        weak["patient_id"] = pid
        strong["patient_id"] = pid

        return {"weak": weak, "strong": strong, "patient_id": pid}

# ---------- public API ----------
def prepare_semi_supervised(
    in_dir: str,
    *,
    cache: bool = False,
    cache_rate: float = 1.0,
    num_workers: int = 4,
    batch_size_labeled: int = 1,
    batch_size_unlabeled: int = 2,
    batch_size_val: int = 1,
    labeled_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    unlabeled_weak_transform: Optional[Callable] = None,
    unlabeled_strong_transform: Optional[Callable] = None,
    seed: int = 0,
):
    """
    Build dataloaders for semi-supervised training using the folder layout:

      {in_dir}/
        TrainVolume/{t2w, adc, hbv}
        ValidVolume/{t2w, adc, hbv}
        UnlabeledVolume/{t2w, adc, hbv}
        Binary_TrainSegmentation/
        Binary_ValidSegmentation/

    Returns:
        labeled_loader, unlabeled_loader, val_loader
    """
    set_determinism(seed=seed)

    # Collect samples
    labeled_samples   = _collect_labeled(in_dir)
    val_samples       = _collect_val(in_dir)
    unlabeled_samples = _collect_unlabeled(in_dir)

    # Transforms
    labeled_transform         = labeled_transform or get_train_transforms()
    val_transform             = val_transform or get_test_transforms()
    unlabeled_weak_transform  = unlabeled_weak_transform or get_unlabeled_weak_transforms()
    unlabeled_strong_transform= unlabeled_strong_transform or get_unlabeled_strong_transforms()

    # Datasets
    if cache:
        labeled_ds = CacheDataset(labeled_samples, transform=labeled_transform, cache_rate=cache_rate, num_workers=num_workers)
        val_ds     = CacheDataset(val_samples,     transform=val_transform,     cache_rate=cache_rate, num_workers=num_workers)
    else:
        labeled_ds = Dataset(labeled_samples, transform=labeled_transform)
        val_ds     = Dataset(val_samples,     transform=val_transform)

    unlabeled_dual_ds = DualViewUnlabeledDataset(
        data=unlabeled_samples,
        weak_transform=unlabeled_weak_transform,
        strong_transform=unlabeled_strong_transform,
        cache=cache,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )

    # Loaders
    labeled_loader = DataLoader(
        labeled_ds, batch_size=batch_size_labeled, shuffle=True, num_workers=num_workers
    )
    unlabeled_loader = DataLoader(
        unlabeled_dual_ds, batch_size=batch_size_unlabeled, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size_val, shuffle=False, num_workers=num_workers
    )

    print(f"[SemiSup] Labeled: {len(labeled_ds)}, Unlabeled: {len(unlabeled_dual_ds)}, Val: {len(val_ds)}")
    return labeled_loader, unlabeled_loader, val_loader
