# src/dataloaders/check.py
import os, sys
# Add the repo root (…\sage_ssl_repo (1)) to sys.path so "src.*" imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.dataloaders.semisup_loader import prepare_semi_supervised  # absolute import

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
    print("[OK] Labeled:", len(labeled_loader.dataset),
          "Unlabeled:", len(unlabeled_loader.dataset),
          "Val:", len(val_loader.dataset))

if __name__ == "__main__":
    main()
