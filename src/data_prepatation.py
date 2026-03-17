
import numpy as np
import pandas as pd
from pathlib import Path

sleep_stage_cols = ['sleep_stage', 'sleep_stage_transition', 'sleep_stage_trans_prop']
apnea_cols = ['apnea_obstructive', 'apnea_central', 'apnea_hypopnea', 'apnea_mixed']
epoch_cols = ['epoch_id', 'epoch_start', 'epoch_end']
subject_col = []
additional_cols = ['ahi', 'oahi', 'arousal_index']


def create_batched_splits(
    parquet_files: list,
    batch_size: int = 360,
    gap_size: int = 6,
    top_features: list[str] | None = None,
    top_features_lag: int = 5,
    target_type: str = 'apnea',
    target_future_steps: int = 5,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    n_leave_out: int = 5,
    random_seed: int = 2542,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list]:
    """
    Batch parquet files into non-overlapping windows and split into train/val/test sets.

    Args:
        parquet_files:        List of parquet file paths.
        batch_size:           Number of rows per batch.
        gap_size:             Number of rows to skip between batches (to avoid leakage).
        top_features:         Features to create lag features for. None means no lag features.
        top_features_lag:     Number of lag steps to generate for top_features.
        target_type:          Type of target to generate ('apnea' supported).
        target_future_steps:  How many steps ahead the target looks.
        val_ratio:            Fraction of each subject's batches for validation.
        test_ratio:           Fraction of each subject's batches for testing.
        n_leave_out:          Number of subjects to exclude entirely.
        random_seed:          Random seed for reproducibility.

    Returns:
        train_X, train_y, val_X, val_y, test_X, test_y, left_out_subjects
    """
    rng = np.random.default_rng(random_seed)

    batches = _chunk_subjects(
        parquet_files=parquet_files,
        batch_size=batch_size,
        gap_size=gap_size,
        top_features=top_features or [],
        top_features_lag=top_features_lag,
        target_type=target_type,
        target_future_steps=target_future_steps,
    )

    # Drop subjects that produced zero valid batches (recording too short for batch_size)
    batches = {sid: b for sid, b in batches.items() if len(b) > 0}

    subject_ids = np.array(list(batches.keys()))
    if len(subject_ids) == 0:
        raise ValueError(
            "No valid batches were produced. Check that parquet_files is non-empty "
            f"and that recordings are longer than batch_size={batch_size} rows."
        )

    n_leave_out = min(n_leave_out, max(0, len(subject_ids) - 1))  # need at least 1 subject for training
    left_out_subjects = rng.choice(subject_ids, size=n_leave_out, replace=False).tolist()
    print(f"Subjects left out: {left_out_subjects}")

    train_batches, val_batches, test_batches = [], [], []

    for subject_id, subject_batches in batches.items():
        if subject_id in left_out_subjects:
            continue
        # Keep batches in chronological order: train = early night, test = late night.
        # Shuffling would allow future data to leak into training via lag features.
        n = len(subject_batches)
        n_test = round(n * test_ratio)
        n_val = round(n * val_ratio)
        n_train = n - n_val - n_test

        # Interleaved 3:1:1 split: vectorised mod on full groups, then sequential remainder
        n_full  = (n // 5) * 5
        idx     = np.arange(n_full)
        mod     = idx % 5
        train_idx = idx[mod <= 2].tolist()
        val_idx   = idx[mod == 3].tolist()
        test_idx  = idx[mod == 4].tolist()

        n_tr_rem = n_train - len(train_idx)
        n_vl_rem = n_val   - len(val_idx)
        rem = list(range(n_full, n))
        train_idx.extend(rem[:n_tr_rem])
        val_idx.extend(rem[n_tr_rem : n_tr_rem + n_vl_rem])
        test_idx.extend(rem[n_tr_rem + n_vl_rem :])

        train_batches.extend(subject_batches[i] for i in train_idx)
        val_batches.extend(subject_batches[i]   for i in val_idx)
        test_batches.extend(subject_batches[i]  for i in test_idx)

    train_df = pd.concat(train_batches, ignore_index=True)
    val_df = pd.concat(val_batches, ignore_index=True)
    test_df = pd.concat(test_batches, ignore_index=True)

    train_y, train_X = train_df['target'], train_df.drop(columns='target')
    val_y,   val_X   = val_df['target'],   val_df.drop(columns='target')
    test_y,  test_X  = test_df['target'],  test_df.drop(columns='target')

    return train_X, train_y, val_X, val_y, test_X, test_y, left_out_subjects


def _chunk_subjects(
    parquet_files: list,
    batch_size: int,
    gap_size: int,
    top_features: list[str],
    top_features_lag: int,
    target_type: str = 'apnea',
    target_future_steps: int = 5,
) -> dict[str, list[pd.DataFrame]]:
    """
    Read parquet files and chunk each subject's data into non-overlapping batches
    separated by a gap to prevent leakage between splits.

    Args:
        parquet_files:        List of parquet file paths.
        batch_size:           Number of rows per batch.
        gap_size:             Number of rows to skip between consecutive batches.
        top_features:         Features to create lag features for.
        top_features_lag:     Number of lag steps to generate for top_features.
        target_type:          Type of target to generate.
        target_future_steps:  How many steps ahead the target looks.

    Returns:
        Dictionary mapping subject_id -> list of batch DataFrames.
    """
    batches = {}

    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)
        subject_id = Path(parquet_file).stem.split("_")[0]

        subject_batches = []
        for start in range(0, len(df), batch_size + gap_size):
            batch = df.iloc[start : start + batch_size].copy()

            if len(batch) < batch_size:  # drop incomplete trailing batches
                continue

            batch = add_future_target(df=batch, target_type=target_type, future_steps=target_future_steps)
            batch = batch.dropna(subset=['target'])
            batch = add_lag_features(batch, features=top_features, max_lag=top_features_lag)
            batch = remove_feature_columns(batch, target_type)

            batch['subject_id'] = subject_id  # keep track of subject in each batch
            batch['chunk_id'] = f"{subject_id}_chunk{start // (batch_size + gap_size)}"  # unique chunk identifier

            subject_batches.append(batch)

        batches[subject_id] = subject_batches

    return batches


def add_future_target(df: pd.DataFrame, target_type: str = 'apnea', future_steps: int = 0) -> pd.DataFrame:
    """
    Add a future target column to the DataFrame by shifting the target columns forward.

    Args:
        df:           Input DataFrame.
        target_type:  Type of target to generate ('apnea' supported).
        future_steps: How many steps ahead the target looks.

    Returns:
        DataFrame with 'target' column added.
    """
    if target_type == 'apnea':
        df['target'] = df[apnea_cols].shift(-future_steps).max(axis=1)
    elif target_type == 'apnea_type':
        shifted = df[apnea_cols].shift(-future_steps)
        df['target'] = shifted.apply(
            lambda row: 'none'         if row.sum() == 0 else
                        'mixed'        if row.sum() > 1  else
                        'obstructive'  if row['apnea_obstructive'] == 1 else
                        'central'      if row['apnea_central'] == 1 else
                        'hypopnea'     if row['apnea_hypopnea'] == 1 else
                        'mixed',
            axis=1
        )

    elif target_type == 'sleep_stage':
        raise NotImplementedError("Future target for sleep stage prediction is not implemented yet.")

    return df


def remove_feature_columns(df: pd.DataFrame, target_type: str) -> pd.DataFrame:
    """
    Drop non-feature columns from the DataFrame, retaining only model inputs and target.

    Args:
        df:          Input DataFrame.
        target_type: Determines which column groups to drop.

    Returns:
        DataFrame with non-feature columns removed.
    """
    if target_type == 'apnea' or target_type == 'apnea_type':
        columns_to_drop = sleep_stage_cols + apnea_cols + epoch_cols + subject_col + additional_cols
    elif target_type == 'sleep_stage':
        raise NotImplementedError("Feature column removal for sleep stage prediction is not implemented yet.")

    return df.drop(columns=columns_to_drop)


# def add_lag_features(
#     df: pd.DataFrame,
#     features: list[str] | None = None,
#     max_lag: int = 0,
# ) -> pd.DataFrame:
#     """
#     Add lag features for specified columns.

#     Args:
#         df:       Input DataFrame.
#         features: List of feature names to create lag features for.
#         max_lag:  Maximum number of lag periods to create.

#     Returns:
#         DataFrame with lag features added.
#     """
#     if not features or max_lag < 1:
#         return df

#     df = df.copy()
#     for feature in features:
#         for lag in range(1, max_lag + 1):
#             df[f"{feature}_lag{lag}"] = df[feature].shift(lag)

#     # Drop first max_lag rows which have NaN values from the lag shift
#     return df.iloc[max_lag:].reset_index(drop=True)

def add_lag_features(
    df: pd.DataFrame,
    features: list[str] | None = None,
    max_lag: int = 0,
) -> pd.DataFrame:
    """
    Add lag features for specified columns.
    """
    if not features or max_lag < 1:
        return df

    base = df.copy()

    # Build all lagged columns first, then concat once to avoid fragmentation.
    lagged_cols = {
        f"{feature}_lag{lag}": base[feature].shift(lag)
        for feature in features
        for lag in range(1, max_lag + 1)
    }
    lagged_df = pd.DataFrame(lagged_cols, index=base.index)

    out = pd.concat([base, lagged_df], axis=1)

    # Drop first max_lag rows which have NaN values from lag shift
    return out.iloc[max_lag:].reset_index(drop=True)
