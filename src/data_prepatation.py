
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

sleep_stage_cols = ['sleep_stage', 'sleep_stage_transition', 'sleep_stage_trans_prop']
apnea_cols = ['apnea_obstructive', 'apnea_central', 'apnea_hypopnea', 'apnea_mixed']
subject_col = []
additional_cols = ['ahi', 'oahi', 'arousal_index']


def create_train_val_test_splits(
    parquet_files: list,
    metadata: pd.DataFrame,
    top_features: list[str] | None = None,
    top_features_lag: int = 5,
    target_type: str = 'apnea',
    target_future_steps: int = 5,
    val_ratio: float = 0.15,
    test_ratio: float = 0.2,
    n_splits: int = 5,
    random_seed: int = 2542,
):
    subject_df = _process_subjects(
        parquet_files=parquet_files,
        top_features=top_features or [],
        top_features_lag=top_features_lag,
        target_type=target_type,
        target_future_steps=target_future_steps,
    )

    subject_ids = np.array(list(subject_df.keys()))
    if len(subject_ids) == 0:
        raise ValueError("No valid batches were produced. Check that parquet_files is non-empty.")

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    # --- Look up per-subject apnea severity stratum for stratified splitting ---
    severity_lookup = metadata.set_index('subject_id')['ahi_stratum']
    subject_strata = np.array([severity_lookup[sid] for sid in subject_ids])

    # # --- Stratified subject-level splits ---
    # train_subjects, test_subjects = train_test_split(
    #     subject_ids, test_size=test_ratio, random_state=random_seed, stratify=subject_strata
    # )
    # train_strata = subject_strata[np.isin(subject_ids, train_subjects)]
    # train_subjects, val_subjects = train_test_split(
    #     train_subjects, test_size=val_ratio, random_state=random_seed, stratify=train_strata
    # )

    strata_lookup = dict(zip(subject_ids, subject_strata))
    train_subjects, test_subjects = train_test_split(
        subject_ids, test_size=test_ratio, random_state=random_seed, stratify=subject_strata
    )
    train_strata = np.array([strata_lookup[sid] for sid in train_subjects])
    train_subjects, val_subjects = train_test_split(
        train_subjects, test_size=val_ratio, random_state=random_seed, stratify=train_strata
    )

    if n_splits > len(train_subjects):
        raise ValueError(
            f"n_splits={n_splits} cannot be greater than number of train subjects="
            f"{len(train_subjects)}. Reduce n_splits or test_ratio."
        )

    train_df = pd.concat([subject_df[sid] for sid in train_subjects], ignore_index=True)
    val_df = pd.concat([subject_df[sid] for sid in val_subjects], ignore_index=True)
    test_df = pd.concat([subject_df[sid] for sid in test_subjects], ignore_index=True)

    train_df, val_df, test_df = _remove_missing_values(train_df, val_df, test_df, missing_threshold=0.2)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # --- Stratified KFold CV on train subjects ---
    train_subject_strata = subject_strata[np.isin(subject_ids, train_subjects)]
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    row_indices_by_subject = {
        sid: train_df.index[train_df['subject_id'] == sid].to_numpy()
        for sid in train_subjects
    }
    fold_indices = []
    for train_sid_idx, val_sid_idx in kfold.split(train_subjects, train_subject_strata):
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
    # Drop columns that are 100% NaN for any single subject in any split
    all_df = pd.concat([train_df, val_df, test_df])
    per_subject_all_nan = (
        all_df.groupby('subject_id')
        .apply(lambda x: x.isna().all())
        .any(axis=0)
    )
    subject_nan_cols = per_subject_all_nan[per_subject_all_nan].index.tolist()

    # Drop columns with >20% missing across the split
    missing_train = train_df.isnull().mean()
    missing_val   = val_df.isnull().mean()
    missing_test  = test_df.isnull().mean()
    threshold_cols = (
        missing_train[missing_train > missing_threshold].index.tolist() +
        missing_val[missing_val   > missing_threshold].index.tolist() +
        missing_test[missing_test > missing_threshold].index.tolist()
    )

    columns_to_drop = list(set(subject_nan_cols + threshold_cols))
    train_df = train_df.drop(columns=columns_to_drop)
    val_df   = val_df.drop(columns=columns_to_drop)
    test_df  = test_df.drop(columns=columns_to_drop)

    # Drop remaining rows with any NaN
    train_df = train_df[~train_df.isna().any(axis=1)].reset_index(drop=True)
    val_df   = val_df[~val_df.isna().any(axis=1)].reset_index(drop=True)
    test_df  = test_df[~test_df.isna().any(axis=1)].reset_index(drop=True)

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

        if len(df) == 0:
            print(f"⚠ Subject {subject_id} produced 0 rows after processing — skipped.")
            continue
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
