
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split

sleep_stage_cols = ['sleep_stage', 'sleep_stage_transition', 'sleep_stage_trans_prop']
apnea_cols = ['apnea_obstructive', 'apnea_central', 'apnea_hypopnea', 'apnea_mixed']
subject_col = []
additional_cols = ['ahi', 'oahi', 'arousal_index']


def create_train_val_test_splits(
    parquet_files: list,
    top_features: list[str] | None = None,
    top_features_lag: int = 5,
    target_type: str = 'apnea',
    target_future_steps: int = 5,
    val_ratio: float = 0.15,
    test_ratio: float = 0.2,
    n_splits: int = 10,
    random_seed: int = 2542,
):
    """
    Batch parquet files into non-overlapping windows and split into train/val/test sets.

    Args:
        parquet_files:        List of parquet file paths.
        top_features:         Features to create lag features for. None means no lag features.
        top_features_lag:     Number of lag steps to generate for top_features.
        target_type:          Type of target to generate ('apnea' supported).
        target_future_steps:  How many steps ahead the target looks.
        val_ratio:            Fraction of each subject's batches for validation.
        test_ratio:           Fraction of each subject's batches for testing.
        n_splits:             Number of K-fold splits over train subjects.
        random_seed:          Random seed for reproducibility.

    Returns:
        train_X, train_y, fold_indices, val_X, val_y, test_X, test_y
        where fold_indices contains (train_idx, val_idx) for each fold.
    """
    subject_df = _process_subjects(
        parquet_files=parquet_files,
        top_features=top_features or [],
        top_features_lag=top_features_lag,
        target_type=target_type,
        target_future_steps=target_future_steps,
    )

    # Drop subjects that produced zero valid batches (recording too short for batch_size)
    # batches = {sid: b for sid, b in batches.items() if len(b) > 0}

    subject_ids = np.array(list(subject_df.keys()))
    if len(subject_ids) == 0:
        raise ValueError(
            "No valid batches were produced. Check that parquet_files is non-empty "
        )

    # train/test 80/20 split subjects
    train_subjects, test_subjects = train_test_split(subject_ids, test_size=test_ratio, random_state=random_seed)
    train_subjects, val_subjects = train_test_split(train_subjects, test_size=val_ratio, random_state=random_seed)

    train_df = pd.concat([subject_df[sid] for sid in train_subjects], ignore_index=True)
    val_df = pd.concat([subject_df[sid] for sid in val_subjects], ignore_index=True)
    test_df = pd.concat([subject_df[sid] for sid in test_subjects], ignore_index=True)


    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if n_splits > len(train_subjects):
        raise ValueError(
            f"n_splits={n_splits} cannot be greater than number of train subjects="
            f"{len(train_subjects)}. Reduce n_splits or test_ratio."
        )

    train_df, val_df, test_df = _remove_missing_values(train_df, val_df, test_df, missing_threshold=0.2)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Build subject-level CV splits and map each split to row indices in train_df.
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    row_indices_by_subject = {
        sid: train_df.index[train_df['subject_id'] == sid].to_numpy()
        for sid in train_subjects
    }
    fold_indices = []
    for train_sid_idx, val_sid_idx in kfold.split(train_subjects):
        train_sid_fold = train_subjects[train_sid_idx]
        val_sid_fold = train_subjects[val_sid_idx]
        train_idx = np.concatenate([row_indices_by_subject[sid] for sid in train_sid_fold])
        val_idx = np.concatenate([row_indices_by_subject[sid] for sid in val_sid_fold])
        fold_indices.append((train_idx, val_idx))

    train_y = train_df['target']
    train_X = train_df.drop(columns='target')
    val_y = val_df['target']
    val_X = val_df.drop(columns='target')
    test_y = test_df['target']
    test_X = test_df.drop(columns='target')

    return train_X, train_y, fold_indices, val_X, val_y, test_X, test_y

def _remove_missing_values(train_df, val_df, test_df, missing_threshold=0.2):
    missing_train = train_df.isnull().mean()
    missing_val = val_df.isnull().mean()
    missing_test = test_df.isnull().mean()

    # make a list of columns which have more than 20% missing values in any of the three sets
    columns_to_drop = missing_train[missing_train > missing_threshold].index.tolist() + \
        missing_val[missing_val > missing_threshold].index.tolist() + \
        missing_test[missing_test > missing_threshold].index.tolist()
    columns_to_drop = list(set(columns_to_drop))
    train_df = train_df.drop(columns=columns_to_drop)
    val_df = val_df.drop(columns=columns_to_drop)
    test_df = test_df.drop(columns=columns_to_drop)

    # find rows with NaN values in train_X, train_y, val_X, val_y, test_X, test_y
    train_nan_rows = train_df.isna().any(axis=1)
    val_nan_rows = val_df.isna().any(axis=1)
    test_nan_rows = test_df.isna().any(axis=1)
    # remove rows with NaN values
    train_df = train_df[~train_nan_rows].reset_index(drop=True)
    val_df = val_df[~val_nan_rows].reset_index(drop=True)
    test_df = test_df[~test_nan_rows].reset_index(drop=True)
    return train_df, val_df, test_df

def _process_subjects(
    parquet_files: list,
    top_features: list[str],
    top_features_lag: int,
    target_type: str = 'apnea',
    target_future_steps: int = 5,
) -> dict[str, pd.DataFrame]:
    """
    Read parquet files and chunk each subject's data into non-overlapping batches
    separated by a gap to prevent leakage between splits.

    Args:
        parquet_files:        List of parquet file paths.
        top_features:         Features to create lag features for.
        top_features_lag:     Number of lag steps to generate for top_features.
        target_type:          Type of target to generate.
        target_future_steps:  How many steps ahead the target looks.

    Returns:
        Dictionary mapping subject_id -> processed DataFrame.
    """
    subject_df = {}
    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)
        subject_id = Path(parquet_file).stem.split("_")[0]

        df = add_future_target(df=df, target_type=target_type, future_steps=target_future_steps)
        df = df.dropna(subset=['target'])
        df = add_lag_features(df, features=top_features, max_lag=top_features_lag)
        df = remove_feature_columns(df, target_type)

        df['subject_id'] = subject_id  # keep track of subject in each batch
        subject_df[subject_id] = df

    return subject_df


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
        columns_to_drop = sleep_stage_cols + apnea_cols + subject_col + additional_cols
    elif target_type == 'sleep_stage':
        raise NotImplementedError("Feature column removal for sleep stage prediction is not implemented yet.")
    else:
        raise ValueError(f"Unsupported target_type: {target_type}")

    return df.drop(columns=columns_to_drop)



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
