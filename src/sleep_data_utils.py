from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
import json
from math import gcd
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch


DEFAULT_STAGE_MAP = {
    "W": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "R": 4,
}


@dataclass
class BlockSample:
    subject_id: str
    block_id: int
    start_idx: int
    end_idx: int


def load_sleep_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def ratio_to_tag(ratio: Sequence[float]) -> str:
    parts = []
    for value in ratio:
        parts.append(str(int(round(float(value) * 100))))
    return "_".join(parts)


def render_subject_split_file_template(
    template: str,
    *,
    seed: int,
    ratio: Sequence[float],
) -> str:
    ratio_tag = ratio_to_tag(ratio)
    return template.format(seed=seed, ratio_tag=ratio_tag)


def build_subject_list(
    prefix: str,
    start: int,
    end: int,
    *,
    small_batch_test: bool = False,
    small_batch_limit: int = 35,
) -> list[str]:
    subjects = [f"{prefix}{n:03d}" for n in range(start, end + 1)]
    if small_batch_test:
        subjects = subjects[:small_batch_limit]
    return subjects


class SleepDataset:
    """
    Dataset for contiguous fixed-duration blocks from subject parquet files.
    """

    def __init__(
        self,
        signals_dir: str | Path,
        metadata_csv: str | Path,
        feature_cols: Sequence[str],
        label_cols: Sequence[str],
        subject_col_meta: str = "sid",
        signals_file_pattern: str = "{subject_id}.parquet",
        block_duration_sec: float = 30.0 * 60.0,
        epoch_sec: Optional[float] = 5.0,
        horizon: int = 0,
        allowed_subjects: Optional[Sequence[str]] = None,
        drop_boundary: int = 25 // 5,
        tail_policy: str = "drop",
        preload: bool = False,
        meta_feature_cols: Optional[Sequence[str]] = None,
    ) -> None:
        self.signals_dir = Path(signals_dir)
        self.metadata = pd.read_csv(metadata_csv)
        self.feature_cols = list(feature_cols)
        self.label_cols = list(label_cols)
        self.subject_col_meta = subject_col_meta
        self.signals_file_pattern = signals_file_pattern
        self.preload = preload
        self.meta_feature_cols = list(meta_feature_cols) if meta_feature_cols is not None else []

        if epoch_sec is None or block_duration_sec <= epoch_sec:
            raise ValueError(f"block_duration_sec must be >= epoch_sec {epoch_sec}s")
        self.block_size = int(block_duration_sec // epoch_sec)

        if horizon < 0:
            raise ValueError("horizon must be >= 0.")
        self.horizon = horizon
        self.drop_boundary = drop_boundary + horizon

        policies = {"merge", "drop", "pad"}
        if tail_policy not in policies:
            tail_policy = "drop"
        self.tail_policy = tail_policy

        all_subjects = self.metadata[self.subject_col_meta].astype(str).tolist()
        if not all_subjects:
            raise KeyError(f"No subjects found from {metadata_csv}.")

        if allowed_subjects is not None:
            allowed_set = set(map(str, allowed_subjects))
            all_subjects = [sid for sid in all_subjects if sid in allowed_set]

        self.subject_ids = all_subjects

        self.meta_by_subject = (
            self.metadata.assign(**{self.subject_col_meta: self.metadata[self.subject_col_meta].astype(str)})
            .set_index(self.subject_col_meta)
            .to_dict(orient="index")
        )

        self._cache: dict[str, pd.DataFrame] = {}
        if self.preload:
            for sid in self.subject_ids:
                self._cache[sid] = self._load_subject_df(sid)

        self.all_blocks: list[BlockSample] = self._build_index()

    def _subject_path(self, subject_id: str) -> Path:
        return self.signals_dir / self.signals_file_pattern.format(subject_id=subject_id)

    def _load_subject_df(self, subject_id: str) -> pd.DataFrame:
        path = self._subject_path(subject_id)
        if not path.exists():
            raise FileNotFoundError(f"Missing parquet for subject {subject_id}: {path}")
        df = pd.read_parquet(path).copy()

        required = set(self.feature_cols + self.label_cols)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} file missing columns: {sorted(missing)}")
        return df

    def _get_subject_df(self, subject_id: str) -> pd.DataFrame:
        if subject_id not in self._cache:
            self._cache[subject_id] = self._load_subject_df(subject_id)
        return self._cache[subject_id]

    def _build_index(self) -> list[BlockSample]:
        all_blocks: list[BlockSample] = []
        needed_block_len = 2 * self.drop_boundary
        if self.block_size < needed_block_len:
            raise ValueError(
                f"block_size={self.block_size} is too small for drop_boundary={needed_block_len}"
            )

        for sid in self.subject_ids:
            df = self._get_subject_df(sid)
            n = len(df)
            n_blocks = n // self.block_size
            for b in range(n_blocks):
                start = b * self.block_size
                end = start + self.block_size
                all_blocks.append(
                    BlockSample(
                        subject_id=sid,
                        block_id=b,
                        start_idx=start,
                        end_idx=end,
                    )
                )
        return all_blocks

    def __len__(self) -> int:
        return len(self.all_blocks)

    def __getitem__(self, idx: int):
        block = self.all_blocks[idx]
        df = self._get_subject_df(block.subject_id)
        block_df = df.iloc[block.start_idx:block.end_idx].reset_index(drop=True)

        if self.drop_boundary > 0:
            block_df = block_df.iloc[self.drop_boundary:len(block_df)].reset_index(drop=True)

        X = block_df[self.feature_cols].to_numpy(dtype=np.float32)
        y = block_df[self.label_cols].to_numpy()

        if self.horizon > 0:
            if len(block_df) <= self.horizon:
                raise ValueError(
                    f"Block sid={block.subject_id}, block_id={block.block_id} is too short for horizon={self.horizon}."
                )
            X = X[:-self.horizon]
            y = y[self.horizon:]

        if self.meta_feature_cols:
            subject_meta = self.meta_by_subject.get(block.subject_id, {})
            meta_values = np.array(
                [subject_meta.get(col, np.nan) for col in self.meta_feature_cols],
                dtype=np.float32,
            )
            meta_matrix = np.tile(meta_values, (X.shape[0], 1))
            X = np.hstack([X, meta_matrix])

        meta = {
            "subject_id": block.subject_id,
            "block_id": block.block_id,
            "start_idx": block.start_idx,
            "end_idx": block.end_idx,
            "n_rows_raw": block.end_idx - block.start_idx,
            "n_rows_final": len(y),
            "subject_meta": self.meta_by_subject.get(block.subject_id, {}),
        }

        return X, y, meta


def make_leave_n_out_split(
    subject_ids: Sequence[str],
    n_test: int,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    import random

    subject_ids = list(map(str, subject_ids))
    if not (1 <= n_test < len(subject_ids)):
        raise ValueError("n_test must be between 1 and len(subject_ids)-1.")
    rng = random.Random(seed)
    shuffled = subject_ids[:]
    rng.shuffle(shuffled)
    test_subjects = shuffled[:n_test]
    train_subjects = shuffled[n_test:]
    return train_subjects, test_subjects


def split_subjects_by_ratio(
    subject_ids: Sequence[str],
    ratio: Sequence[float] = (0.64, 0.16, 0.2),
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    import random

    if len(ratio) != 3:
        raise ValueError("ratio must have 3 entries: (train, val, test)")

    ratio_sum = float(sum(ratio))
    if ratio_sum <= 0:
        raise ValueError("ratio sum must be positive")

    normalized = [float(r) / ratio_sum for r in ratio]

    subject_ids = list(map(str, subject_ids))
    if len(subject_ids) < 3:
        raise ValueError("Need at least 3 subjects to split into train/val/test")

    rng = random.Random(seed)
    shuffled = subject_ids[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * normalized[0]))
    n_val = int(round(n * normalized[1]))

    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))
    n_test = n - n_train - n_val

    if n_test <= 0:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1

    train_subjects = shuffled[:n_train]
    val_subjects = shuffled[n_train:n_train + n_val]
    test_subjects = shuffled[n_train + n_val:]
    return train_subjects, val_subjects, test_subjects


def load_or_create_subject_split(
    subject_ids: Sequence[str],
    split_file_path: str | Path,
    *,
    ratio: Sequence[float] = (0.64, 0.16, 0.2),
    seed: int = 42,
) -> dict:
    split_file_path = Path(split_file_path)

    if split_file_path.exists():
        with split_file_path.open("r", encoding="utf-8") as fp:
            saved = json.load(fp)
        required = {"train_subjects", "val_subjects", "test_subjects"}
        if not required.issubset(saved.keys()):
            raise ValueError(
                f"Existing split file missing keys {required}: {split_file_path}"
            )
        return saved

    train_subjects, val_subjects, test_subjects = split_subjects_by_ratio(
        subject_ids=subject_ids,
        ratio=ratio,
        seed=seed,
    )

    payload = {
        "seed": int(seed),
        "ratio": [float(r) for r in ratio],
        "num_subjects": len(subject_ids),
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "test_subjects": test_subjects,
    }

    split_file_path.parent.mkdir(parents=True, exist_ok=True)
    with split_file_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    return payload


def split_within_subject(
    dataset: SleepDataset,
    ratio: Sequence[float] = (0.6, 0.2, 0.2),
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]] | None:
    import random

    subject_blocks = defaultdict(list)
    for idx, block in enumerate(dataset.all_blocks):
        subject_blocks[block.subject_id].append(idx)

    train_ids: list[int] = []
    val_ids: list[int] = []
    test_ids: list[int] = []

    for subject_id, indices in subject_blocks.items():
        indices = sorted(indices, key=lambda i: dataset.all_blocks[i].block_id)
        subject_seed = seed + int(subject_id[1:]) * 2
        rng = random.Random(subject_seed)
        indices = indices[:]
        rng.shuffle(indices)

        n = len(indices)
        if n < 3:
            return None

        n_train = round(n * ratio[0])
        n_val = round(n * ratio[1])

        train_ids.extend(indices[:n_train])
        val_ids.extend(indices[n_train:n_train + n_val])
        test_ids.extend(indices[n_train + n_val:])

    return train_ids, val_ids, test_ids


def split_within_subject_ordered(
    dataset: SleepDataset,
    ratio: Sequence[float] = (0.6, 0.2, 0.2),
) -> tuple[list[int], list[int], list[int]]:
    subject_blocks = defaultdict(list)
    for idx, block in enumerate(dataset.all_blocks):
        subject_blocks[block.subject_id].append(idx)

    train_ids: list[int] = []
    val_ids: list[int] = []
    test_ids: list[int] = []

    denom = reduce(gcd, [int(r * 100) for r in ratio])
    pattern = [int(r * 100 / denom) for r in ratio]
    pattern_seq = ["train"] * pattern[0] + ["val"] * pattern[1] + ["test"] * pattern[2]
    L = len(pattern_seq)

    for subject_id, indices in subject_blocks.items():
        indices = sorted(indices, key=lambda i: dataset.all_blocks[i].block_id)
        if len(indices) < 3:
            continue

        for k, idx in enumerate(indices):
            tag = pattern_seq[k % L]
            if tag == "train":
                train_ids.append(idx)
            elif tag == "val":
                val_ids.append(idx)
            else:
                test_ids.append(idx)

    return train_ids, val_ids, test_ids


class DatasetSubset:
    def __init__(self, dataset: SleepDataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        return self.dataset[self.indices[i]]


def flatten_dataset(dataset: SleepDataset | DatasetSubset):
    X_list = []
    y_list = []
    rows = []

    for i in range(len(dataset)):
        X_block, y_block, meta = dataset[i]
        X_list.append(X_block)
        y_list.append(y_block)
        rows.extend(
            {
                "subject_id": meta["subject_id"],
                "block_id": meta["block_id"],
                "row_in_block": row_idx,
            }
            for row_idx in range(len(y_block))
        )

    X = np.vstack(X_list)
    y = np.vstack(y_list)
    info = pd.DataFrame(rows)
    return X, y, info


def encode_labels(X, y, info=None, stage_map: Optional[dict] = None):
    if stage_map is None:
        stage_map = DEFAULT_STAGE_MAP

    y_arr = np.asarray(y).reshape(-1)
    X_arr = np.asarray(X)
    mask = np.isin(y_arr, list(stage_map.keys()))
    X_arr = X_arr[mask]
    y_arr = y_arr[mask]
    y_arr = np.array([stage_map[s] for s in y_arr], dtype=np.int64)

    if info is None:
        return X_arr, y_arr

    info_df = info if isinstance(info, pd.DataFrame) else pd.DataFrame(info)
    if len(info_df) != len(mask):
        raise ValueError(
            f"info length {len(info_df)} must match y length {len(mask)} for filtering."
        )
    info_df = info_df.loc[mask].reset_index(drop=True)
    return X_arr, y_arr, info_df


class SleepSequenceDataset(Dataset):
    def __init__(
        self,
        X,
        y,
        info,
        seq_len=30,
        subject_col="subject_id",
        block_col="block_id",
        row_col="row_in_block",
    ):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.info = info.copy() if isinstance(info, pd.DataFrame) else pd.DataFrame(info)
        self.seq_len = seq_len
        self.samples = []

        required_cols = [subject_col, block_col, row_col]
        missing = [c for c in required_cols if c not in self.info.columns]
        if missing:
            raise ValueError(f"Missing columns in info: {missing}")

        if len(self.X) != len(self.y) or len(self.X) != len(self.info):
            raise ValueError(
                f"Length mismatch: len(X)={len(self.X)}, len(y)={len(self.y)}, len(info)={len(self.info)}"
            )

        for i, row_in_block in enumerate(self.info[row_col].to_numpy()):
            if row_in_block >= seq_len - 1:
                self.samples.append(i)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        end_idx = self.samples[i]
        start_idx = end_idx - self.seq_len + 1

        x_seq = torch.from_numpy(self.X[start_idx:end_idx + 1])
        y_target = torch.tensor(self.y[end_idx], dtype=torch.long)
        return x_seq, y_target


def preprocess_train_val_test(
    X_train,
    X_val,
    X_test,
    *,
    imputer_strategy: str = "constant",
    imputer_fill_value: float = 0.0,
    scaler: str = "standard",
):
    imputer = SimpleImputer(strategy=imputer_strategy, fill_value=imputer_fill_value)
    X_train2 = imputer.fit_transform(X_train)
    X_val2 = imputer.transform(X_val)
    X_test2 = imputer.transform(X_test)

    if scaler != "standard":
        raise ValueError(f"Unsupported scaler: {scaler}")

    scaler_obj = StandardScaler()
    X_train_scaled = scaler_obj.fit_transform(X_train2)
    X_val_scaled = scaler_obj.transform(X_val2)
    X_test_scaled = scaler_obj.transform(X_test2)

    return {
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "imputer": imputer,
        "scaler": scaler_obj,
    }