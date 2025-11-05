import os
from glob import glob
import random
from typing import List, Optional, Dict, Callable, Tuple
from monai.data import DataLoader, Dataset, CacheDataset
from torch.utils.data import Dataset as TorchDataset
from monai.utils import set_determinism
from src.training_setup.augmentations.train_augment import get_train_transforms
from src.training_setup.augmentations.test_augment import get_test_transforms
try:
    from src.training_setup.augmentations.unlabeled_augment import get_unlabeled_weak_transforms, get_unlabeled_strong_transforms
except Exception:
    def get_unlabeled_weak_transforms(): return get_train_transforms()
    def get_unlabeled_strong_transforms(): return get_train_transforms()
def _collect_paths(in_dir: str):
    path_train_t2w = sorted(glob(os.path.join(in_dir, "TrainVolumes", "t2w", "*.nii.gz")))
    path_train_adc = sorted(glob(os.path.join(in_dir, "TrainVolumes", "adc", "*.nii.gz")))
    path_train_hbv = sorted(glob(os.path.join(in_dir, "TrainVolumes", "hbv", "*.nii.gz")))
    path_train_seg = sorted(glob(os.path.join(in_dir, "Binary_TrainSegmentation", "*.nii.gz")))
    path_val_t2w = sorted(glob(os.path.join(in_dir, "ValidVolumes", "t2w", "*.nii.gz")))
    path_val_adc = sorted(glob(os.path.join(in_dir, "ValidVolumes", "adc", "*.nii.gz")))
    path_val_hbv = sorted(glob(os.path.join(in_dir, "ValidVolumes", "hbv", "*.nii.gz")))
    path_val_seg = sorted(glob(os.path.join(in_dir, "Binary_ValidSegmentation", "*.nii.gz")))
    def pid(p): return os.path.basename(p).split("_")[0]
    train_pids = [pid(p) for p in path_train_t2w]
    val_pids = [pid(p) for p in path_val_t2w]
    train_samples = [ {"patient_id": pid_i, "t2w": t2w, "adc": adc, "hbv": hbv, "seg": seg}
        for pid_i, t2w, adc, hbv, seg in zip(train_pids, path_train_t2w, path_train_adc, path_train_hbv, path_train_seg) ]
    val_samples = [ {"patient_id": pid_i, "t2w": t2w, "adc": adc, "hbv": hbv, "seg": seg}
        for pid_i, t2w, adc, hbv, seg in zip(val_pids, path_val_t2w, path_val_adc, path_val_hbv, path_val_seg) ]
    return train_samples, val_samples
def _split_labeled_unlabeled(train_samples: List[Dict], labeled_fraction: float=0.1, labeled_patient_ids: Optional[List[str]]=None, seed: int=0) -> Tuple[List[Dict], List[Dict]]:
    assert 0 < labeled_fraction <= 1.0 or labeled_patient_ids, "Provide labeled_fraction in (0,1] or labeled_patient_ids."
    pid_to_sample = {s["patient_id"]: s for s in train_samples}
    all_pids = list(pid_to_sample.keys())
    if labeled_patient_ids is None:
        rnd = random.Random(seed); rnd.shuffle(all_pids)
        k = max(1, int(round(len(all_pids) * labeled_fraction)))
        labeled_pids = set(all_pids[:k])
    else:
        labeled_pids = set(labeled_patient_ids)
    labeled = [pid_to_sample[pid] for pid in all_pids if pid in labeled_pids]
    unlabeled = [pid_to_sample[pid] for pid in all_pids if pid not in labeled_pids]
    unlabeled = [{k: v for k, v in s.items() if k != "seg"} for s in unlabeled]
    return labeled, unlabeled
class DualViewUnlabeledDataset(TorchDataset):
    def __init__(
        self,
        data: List[Dict],
        weak_transform: Callable,
        strong_transform: Callable,
        cache: bool = False,
        cache_rate: float = 1.0,
    ):
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

        if cache:
            base = CacheDataset
            self.base_ds = base(data=data, transform=None, cache_rate=cache_rate)
        else:
            base = Dataset
            self.base_ds = base(data=data, transform=None)  # <- no cache_rate here

def prepare_semi_supervised(
    in_dir: str,
    cache: bool = False,
    cache_rate: float = 1.0,
    batch_size_labeled: int = 1,
    batch_size_unlabeled: int = 1,
    batch_size_val: int = 1,
    labeled_fraction: float = 0.1,
    labeled_patient_ids: Optional[List[str]] = None,
    seed: int = 0,
    labeled_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    unlabeled_weak_transform: Optional[Callable] = None,
    unlabeled_strong_transform: Optional[Callable] = None,
):
    set_determinism(seed=seed)

    train_samples, val_samples = _collect_paths(in_dir)
    labeled_samples, unlabeled_samples = _split_labeled_unlabeled(
        train_samples, labeled_fraction=labeled_fraction, labeled_patient_ids=labeled_patient_ids, seed=seed
    )

    labeled_transform = labeled_transform or get_train_transforms()
    val_transform = val_transform or get_test_transforms()
    unlabeled_weak_transform = unlabeled_weak_transform or get_unlabeled_weak_transforms()
    unlabeled_strong_transform = unlabeled_strong_transform or get_unlabeled_strong_transforms()

    # --- Build labeled & val datasets with the correct class/args ---
    if cache:
        labeled_ds = CacheDataset(data=labeled_samples, transform=labeled_transform, cache_rate=cache_rate)
        val_ds     = CacheDataset(data=val_samples,     transform=val_transform,     cache_rate=cache_rate)
    else:
        labeled_ds = Dataset(data=labeled_samples, transform=labeled_transform)
        val_ds     = Dataset(data=val_samples,     transform=val_transform)

    # --- Unlabeled dual-view dataset already handles cache vs. non-cache internally ---
    unlabeled_dual_ds = DualViewUnlabeledDataset(
        data=unlabeled_samples,
        weak_transform=unlabeled_weak_transform,
        strong_transform=unlabeled_strong_transform,
        cache=cache,
        cache_rate=cache_rate,
    )

    labeled_loader   = DataLoader(labeled_ds,   batch_size=batch_size_labeled,   shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_dual_ds, batch_size=batch_size_unlabeled, shuffle=True)
    val_loader       = DataLoader(val_ds,       batch_size=batch_size_val,       shuffle=False)

    print(f"[SemiSup] Labeled: {len(labeled_ds)}, Unlabeled: {len(unlabeled_dual_ds)}, Val: {len(val_ds)}")
    return labeled_loader, unlabeled_loader, val_loader
