import datasets
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from transformers import DataCollatorForLanguageModeling


def create_calibration_dataloader(
    calibration_dataset_path,
    num_calibration_samples,
    train_dataset, 
    tokenizer
):
    """
    Create a calibration DataLoader for Wanda-based freezing strategies.
    
    Args:
        calibration_dataset_path (str): Path to the calibration dataset. If None, uses a subset of the training dataset.
        num_calibration_samples (int): Number of samples to use for calibration.
        train_dataset: The training dataset
        tokenizer: The tokenizer
    
    Returns:
        DataLoader: Calibration data loader, or None if not needed
    """
    if calibration_dataset_path is not None:
        # Load separate calibration dataset
        calibration_dataset = datasets.load_from_disk(calibration_dataset_path)
        print(f"Loaded calibration dataset from {calibration_dataset_path}")
    else:
        # Use a subset of the training dataset for calibration
        calibration_dataset = train_dataset
        print("Using subset of training dataset for calibration")
    
    # Create a subset for calibration
    total_samples = len(calibration_dataset)
    num_samples = min(num_calibration_samples, total_samples)
    
    # Use deterministic sampling for reproducibility
    indices = list(range(0, total_samples, max(1, total_samples // num_samples)))[:num_samples]
    calibration_subset = Subset(calibration_dataset, indices)
    
    print(f"Created calibration subset with {len(calibration_subset)} samples")
    
    # Create data collator for calibration
    calibration_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        return_tensors="pt"
    )
    
    # Create calibration dataloader
    calibration_dataloader = DataLoader(
        calibration_subset,
        batch_size=1,  # Process one sample at a time for activation collection
        shuffle=False,  # Deterministic for reproducibility
        collate_fn=calibration_collator,
        pin_memory=torch.cuda.is_available(),
        num_workers=0  # Avoid multiprocessing issues with hooks
    )
    
    return calibration_dataloader



import random


def build_replay_subset(dataset, num_samples: int, seed: int = 42):
    n = len(dataset)
    if n <= 0 or num_samples <= 0:
        return None
    num_samples = min(num_samples, n)
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    return Subset(dataset, idx[:num_samples])

def maybe_concat_replay_datasets(train_dataset, replay_datasets, replay_ratio: float, seed: int):
    """
    TRACE-style: 从旧数据里抽子集，拼进当前数据。避免 datasets.shuffle/select 写临时 Arrow 文件。
    replay_ratio 表示“replay 在整体训练数据中的占比近似值”。
    """
    if not replay_datasets or replay_ratio is None or replay_ratio <= 0:
        return train_dataset
    if replay_ratio >= 1.0:
        raise ValueError("replay_ratio must be in (0, 1).")

    target_replay_size = int((replay_ratio / (1.0 - replay_ratio)) * len(train_dataset))
    if target_replay_size <= 0:
        return train_dataset

    per_ds = max(1, target_replay_size // len(replay_datasets))
    subsets = []
    for i, ds in enumerate(replay_datasets):
        sub = build_replay_subset(ds, per_ds, seed + i)
        if sub is not None:
            subsets.append(sub)

    if not subsets:
        return train_dataset

    replay_all = ConcatDataset(subsets)
    return ConcatDataset([train_dataset, replay_all])
