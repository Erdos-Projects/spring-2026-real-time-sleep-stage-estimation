import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_classif

# ── Styling ────────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', font_scale=1.1)
APNEA_PALETTE = {0: '#4C72B0', 1: '#DD8452'}  # blue=no apnea, orange=apnea


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE TYPE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def split_feature_types(
    train_X: pd.DataFrame,
    timeseries_features: list[str] | None = None,
    metadata_features: list[str] | None = None,

) -> tuple[list[str], list[str]]:
    """
    Automatically detect which features are metadata (constant per subject)
    vs time-dependent (vary within a subject's recording).

    Args:
        train_X: Training features DataFrame (must include subject_id).

    Returns:
        metadata_features:   Features that are constant within every subject.
        timeseries_features: Features that vary within at least one subject.
    """
    numeric_cols = train_X.select_dtypes(include=np.number).columns.tolist()

    # Compute std within each subject, then take the mean across subjects
    within_subject_std = (
        train_X[['subject_id'] + numeric_cols]
        .groupby('subject_id')[numeric_cols]
        .std()
        .mean()  # mean of per-subject stds
    )

    metadata_features   = within_subject_std[within_subject_std < 1e-6].index.tolist()
    timeseries_features = within_subject_std[within_subject_std >= 1e-6].index.tolist()

    print(f"Metadata features   (constant per subject): {len(metadata_features)}")
    print(f"Time-series features (vary within subject): {len(timeseries_features)}")
    print(f"\nMetadata features: {metadata_features}")

    return metadata_features, timeseries_features


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Data Quality
# ══════════════════════════════════════════════════════════════════════════════

def plot_missing_values(train_X: pd.DataFrame, threshold: float = 0.01):
    """
    Plot features with missing values above threshold.
    Bars are coloured by missingness severity.
    Pass only time-series features — metadata features are excluded.

    Args:
        train_X:   Training features DataFrame (time-series features only).
        threshold: Only show features with missing rate above this value.
    """
    missing = train_X.isnull().mean().sort_values(ascending=False)
    missing = missing[missing > threshold]

    if missing.empty:
        print("✓ No features with missing values above threshold.")
        return

    colors = ['#d62728' if v > 0.2 else '#ff7f0e' if v > 0.05 else '#4C72B0'
              for v in missing.values]

    fig, ax = plt.subplots(figsize=(12, max(4, len(missing) * 0.3)))
    ax.barh(missing.index, missing.values, color=colors)
    ax.axvline(0.2, color='red',    linestyle='--', linewidth=1, label='20% threshold')
    ax.axvline(0.05, color='orange', linestyle='--', linewidth=1, label='5% threshold')
    ax.set_xlabel('Missing Rate')
    ax.set_title('Missing Values per Feature')
    ax.legend()
    plt.tight_layout()
    plt.show()

    print(f"\nFeatures above 20% missing: {(missing > 0.2).sum()}")
    print(f"Features above  5% missing: {(missing > 0.05).sum()}")


def plot_class_balance(train_y: pd.Series):
    """
    Plot target class distribution and print counts.

    Args:
        train_y: Training target Series.
    """
    counts = train_y.value_counts().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(counts.index.astype(str), counts.values,
                color=[APNEA_PALETTE[i] for i in counts.index])
    axes[0].set_title('Class Counts')
    axes[0].set_xlabel('Target')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 500, f'{v:,}', ha='center', fontsize=10)

    axes[1].pie(counts.values, labels=['No Apnea', 'Apnea'],
                colors=[APNEA_PALETTE[0], APNEA_PALETTE[1]],
                autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Class Balance')

    plt.suptitle('Target Distribution', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print(f"\nNo Apnea : {counts[0]:>8,}  ({counts[0]/len(train_y):.1%})")
    print(f"Apnea    : {counts[1]:>8,}  ({counts[1]/len(train_y):.1%})")


def plot_zero_variance(train_X: pd.DataFrame, timeseries_features: list[str]):
    """
    Print time-series features with zero or near-zero variance (std < 1e-6).
    Metadata features are excluded as they are expected to be constant per subject.

    Args:
        train_X:             Training features DataFrame.
        timeseries_features: Output of split_feature_types() — only these are checked.
    """
    numeric = train_X[timeseries_features].select_dtypes(include=np.number)
    std = numeric.std()
    zero_var = std[std < 1e-6]

    if zero_var.empty:
        print("✓ No zero-variance features found in time-series features.")
    else:
        print(f"⚠ {len(zero_var)} zero-variance features (safe to drop):")
        print(zero_var.index.tolist())

    near_zero = std[(std >= 1e-6) & (std < 0.01)]
    if not near_zero.empty:
        print(f"\n⚠ {len(near_zero)} near-zero variance features (review):")
        print(near_zero.sort_values().head(20))


def analyze_zero_variance_features(
    train_X: pd.DataFrame,
    timeseries_features: list[str],
    std_threshold: float = 1e-6,
    subject_variance_min: float = 0.01,
) -> pd.DataFrame:
    """
    For each zero-variance time-series feature, compute per-subject variance to distinguish:
      - Truly zero-variance: constant across ALL subjects → safe to drop
      - Subject-specific variance: varies in some subjects → investigate further

    Metadata features are excluded as they are expected to be constant per subject.

    Args:
        train_X:               Training features DataFrame (must include subject_id).
        timeseries_features:   Output of split_feature_types() — only these are checked.
        std_threshold:         Global std below which a feature is flagged as zero-variance.
        subject_variance_min:  Per-subject std threshold to count a subject as "varying".

    Returns:
        summary_df: DataFrame with per-feature breakdown.
    """
    numeric_X = train_X[timeseries_features].select_dtypes(include=np.number)
    global_std = numeric_X.std()
    zero_var_features = global_std[global_std < std_threshold].index.tolist()

    if not zero_var_features:
        print("✓ No zero-variance features found in time-series features.")
        return pd.DataFrame()

    print(f"Found {len(zero_var_features)} zero-variance features — checking per-subject variance...\n")

    per_subject_std = (
        train_X[['subject_id'] + zero_var_features]
        .groupby('subject_id')[zero_var_features]
        .std()
    )

    n_subjects_varying = (per_subject_std > subject_variance_min).sum()
    mean_std_varying   = per_subject_std[per_subject_std > subject_variance_min].mean()

    summary_df = pd.DataFrame({
        'feature':             zero_var_features,
        'global_std':          global_std[zero_var_features].values,
        'n_subjects_varying':  n_subjects_varying[zero_var_features].values,
        'mean_std_in_varying': mean_std_varying[zero_var_features].values,
    }).sort_values('n_subjects_varying', ascending=False)

    truly_zero   = summary_df[summary_df['n_subjects_varying'] == 0]
    subject_spec = summary_df[summary_df['n_subjects_varying'] >  0]

    print(f"Truly zero-variance (0 subjects vary) → safe to drop: {len(truly_zero)}")
    print(f"Subject-specific variance              → investigate:  {len(subject_spec)}\n")

    if not subject_spec.empty:
        print("Subject-specific features (sorted by n subjects varying):")
        print(subject_spec.to_string(index=False))

    if not subject_spec.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(subject_spec) * 0.35 + 2)))

        axes[0].barh(subject_spec['feature'][::-1], subject_spec['n_subjects_varying'][::-1],
                     color='#DD8452')
        axes[0].set_title('N Subjects With Real Variance', fontweight='bold')
        axes[0].set_xlabel('Subject count')

        axes[1].barh(subject_spec['feature'][::-1], subject_spec['mean_std_in_varying'][::-1],
                     color='#4C72B0')
        axes[1].set_title('Mean Std (in varying subjects)', fontweight='bold')
        axes[1].set_xlabel('Mean std')

        plt.suptitle('Zero-Variance Features — Per-Subject Breakdown', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.show()

    return summary_df


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Target Relationship
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_distributions(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    features: list[str],
    ncols: int = 3,
):
    """
    KDE plots of each time-series feature split by apnea vs no-apnea.
    Use this to visually identify features with separable distributions.

    Args:
        train_X:  Training features DataFrame.
        train_y:  Training target Series.
        features: List of time-series feature names to plot.
        ncols:    Number of columns in the plot grid.
    """
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5))
    axes = np.array(axes).flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        for label, color in APNEA_PALETTE.items():
            subset = train_X.loc[train_y == label, feat].dropna()
            subset.plot.kde(ax=ax, color=color, label='No Apnea' if label == 0 else 'Apnea',
                            linewidth=2)
        ax.set_title(feat, fontsize=10)
        ax.set_xlabel('')
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Feature Distributions by Target', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

def compute_univariate_scores(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    timeseries_features: list[str],
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Compute point-biserial correlation and mutual information for time-series features only.
    Metadata features are excluded as their relationship to target is at subject level.
    Returns a ranked DataFrame — use this to identify top candidates.

    Args:
        train_X:             Training features DataFrame.
        train_y:             Training target Series.
        timeseries_features: Output of split_feature_types() — only these are scored.
        top_n:               Number of top features to print.

    Returns:
        DataFrame with columns: feature, abs_correlation, mutual_info, combined_rank
    """
    numeric_X = train_X[timeseries_features].select_dtypes(include=np.number).dropna()
    aligned_y = train_y.loc[numeric_X.index]

    print("Computing point-biserial correlations...")
    correlations = {}
    for col in numeric_X.columns:
        corr, _ = stats.pointbiserialr(aligned_y, numeric_X[col])
        correlations[col] = abs(corr)

    print("Computing mutual information scores...")
    mi_scores = mutual_info_classif(numeric_X, aligned_y, random_state=42)

    scores_df = pd.DataFrame({
        'feature':         numeric_X.columns,
        'abs_correlation': list(correlations.values()),
        'mutual_info':     mi_scores,
    })

    scores_df['rank_corr'] = scores_df['abs_correlation'].rank(ascending=False)
    scores_df['rank_mi']   = scores_df['mutual_info'].rank(ascending=False)
    scores_df['combined_rank'] = (scores_df['rank_corr'] + scores_df['rank_mi']) / 2
    scores_df = scores_df.sort_values('combined_rank')

    print(f"\nTop {top_n} features by combined rank:")
    print(scores_df.head(top_n).to_string(index=False))

    return scores_df

def get_redundant_feature_groups(
    train_X: pd.DataFrame,
    timeseries_features: list[str],
    threshold: float = 0.9,
) -> list[list[str]]:
    """
    Cluster time-series features into groups where all members are highly correlated.
    Keep one feature per group, drop the rest.
    Metadata features are excluded as they will always appear correlated at timestep level.

    Args:
        train_X:             Training features DataFrame.
        timeseries_features: Output of split_feature_types().
        threshold:           Correlation threshold to group features.

    Returns:
        List of groups — each group is a list of correlated feature names.
        First element in each group is the suggested feature to keep.
    """
    numeric_X = train_X[timeseries_features].select_dtypes(include=np.number).dropna()
    corr = numeric_X.corr().abs()

    visited = set()
    groups  = []

    for feat in corr.columns:
        if feat in visited:
            continue
        group = [feat]
        visited.add(feat)
        for other in corr.columns:
            if other not in visited and corr.loc[feat, other] >= threshold:
                group.append(other)
                visited.add(other)
        if len(group) > 1:
            groups.append(group)

    print(f"Found {len(groups)} redundant groups (keeping first, dropping rest):")
    features_to_drop = []
    for g in groups:
        print(f"  keep={g[0]:40s}  drop={g[1:]}")
        features_to_drop.extend(g[1:])

    print(f"\nTotal features to drop: {len(features_to_drop)}")
    return groups

def plot_univariate_scores(scores_df: pd.DataFrame, top_n: int = 30):
    """
    Bar plots of top features by correlation and mutual information side by side.

    Args:
        scores_df: Output of compute_univariate_scores().
        top_n:     Number of top features to display.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, top_n * 0.35 + 2))

    top = scores_df.head(top_n)

    axes[0].barh(top['feature'][::-1], top['abs_correlation'][::-1], color='#4C72B0')
    axes[0].set_title('Point-Biserial Correlation (abs)', fontweight='bold')
    axes[0].set_xlabel('|Correlation|')

    axes[1].barh(top['feature'][::-1], top['mutual_info'][::-1], color='#DD8452')
    axes[1].set_title('Mutual Information', fontweight='bold')
    axes[1].set_xlabel('MI Score')

    plt.suptitle(f'Top {top_n} Features — Univariate Scores', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Feature Redundancy
# ══════════════════════════════════════════════════════════════════════════════

def plot_correlation_matrix(
    train_X: pd.DataFrame,
    timeseries_features: list[str],
    features: list[str] | None = None,
    threshold: float = 0.9,
):
    """
    Heatmap of feature correlations for time-series features only.
    Metadata features are excluded as constant-per-subject features
    will show artificially high correlations at the timestep level.

    Args:
        train_X:             Training features DataFrame.
        timeseries_features: Output of split_feature_types().
        features:            Subset of timeseries_features to plot. None = all.
        threshold:           Correlation threshold to flag as redundant.
    """
    cols = features if features else timeseries_features
    numeric_X = train_X[cols].select_dtypes(include=np.number)
    corr = numeric_X.corr()

    redundant = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if abs(corr.iloc[i, j]) >= threshold:
                redundant.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

    fig, ax = plt.subplots(figsize=(max(10, len(corr) * 0.4), max(8, len(corr) * 0.4)))
    sns.heatmap(corr, ax=ax, cmap='coolwarm', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.3,
                xticklabels=corr.columns, yticklabels=corr.columns)
    ax.set_title(f'Feature Correlation Matrix  (threshold={threshold})', fontweight='bold')
    plt.tight_layout()
    plt.show()

    print(f"\n⚠ {len(redundant)} pairs with |correlation| ≥ {threshold}:")
    for f1, f2, c in sorted(redundant, key=lambda x: abs(x[2]), reverse=True)[:20]:
        print(f"  {f1:40s} ↔  {f2:40s}  r={c:.3f}")


def plot_metadata_by_stratum(
    train_X: pd.DataFrame,
    metadata_features: list[str],
):
    if not metadata_features:
        print("No metadata features to plot.")
        return

    # Deduplicate columns first, then exclude ahi_stratum from features to plot
    train_X_dedup = train_X.loc[:, ~train_X.columns.duplicated()]
    plot_feats = [f for f in metadata_features if f != 'ahi_stratum']

    subject_level = (
        train_X_dedup[['subject_id', 'ahi_stratum'] + plot_feats]
        .drop_duplicates('subject_id')
        .copy()
    )
    subject_level['ahi_stratum'] = subject_level['ahi_stratum'].astype(str)

    # Only plot numeric features
    numeric_feats = [
        f for f in plot_feats
        if pd.api.types.is_numeric_dtype(subject_level[f])
    ]
    skipped = set(plot_feats) - set(numeric_feats)
    if skipped:
        print(f"⚠ Skipping non-numeric metadata features: {skipped}")
    if not numeric_feats:
        print("No numeric metadata features to plot.")
        return

    order = [s for s in ['normal', 'mild', 'moderate', 'severe']
             if s in subject_level['ahi_stratum'].values]

    ncols = 3
    nrows = int(np.ceil(len(numeric_feats) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5))
    axes = np.array(axes).flatten()

    for i, feat in enumerate(numeric_feats):
        ax = axes[i]
        sns.boxplot(data=subject_level, x='ahi_stratum', y=feat,
                    order=order, ax=ax, palette='Blues')
        ax.set_title(feat, fontsize=10)
        ax.set_xlabel('')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Metadata Features by AHI Stratum (subject level)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — Domain-informed Grouping
# ══════════════════════════════════════════════════════════════════════════════

def plot_group_importance(
    scores_df: pd.DataFrame,
    feature_groups: dict[str, list[str]],
):
    """
    Bar plot of mean mutual information score per feature group.
    Reveals which signal types are most predictive overall.

    Args:
        scores_df:      Output of compute_univariate_scores().
        feature_groups: Dict mapping group name → list of feature names.
                        Example: {'respiratory': ['spo2', 'airflow', ...],
                                  'cardiac':     ['hr', 'hrv', ...]}
    """
    group_scores = {}
    for group, feats in feature_groups.items():
        matched = scores_df[scores_df['feature'].isin(feats)]
        if not matched.empty:
            group_scores[group] = matched['mutual_info'].mean()

    if not group_scores:
        print("No features matched the provided groups.")
        return

    group_df = pd.Series(group_scores).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(3, len(group_df) * 0.6)))
    group_df.plot.barh(ax=ax, color='#4C72B0')
    ax.set_title('Mean Mutual Information by Feature Group', fontweight='bold')
    ax.set_xlabel('Mean MI Score')
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE — Run all stages at once
# ══════════════════════════════════════════════════════════════════════════════

def run_full_eda(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    metadata: pd.DataFrame,
    feature_groups: dict[str, list[str]] | None = None,
    top_n: int = 30,
    correlation_threshold: float = 0.9,
):
    """
    Run all 4 EDA stages in sequence.

    Args:
        train_X:               Training features (must include subject_id).
        train_y:               Training target Series.
        metadata:              Metadata DataFrame with subject_id and ahi_stratum columns.
        feature_groups:        Optional dict of group_name → list of time-series feature names.
        top_n:                 Number of top features to show in univariate plots.
        correlation_threshold: Threshold for redundancy detection.

    Returns:
        scores_df:           Univariate scores DataFrame for time-series features.
        metadata_features:   List of detected metadata feature names.
        timeseries_features: List of detected time-series feature names.
    """
    # ── Detect feature types first — used by all subsequent stages ──
    print("=" * 60)
    print("FEATURE TYPE DETECTION")
    print("=" * 60)

    # metadata_features are defined as columns present in train_X that are also in metadata
    metadata_features = [col for col in train_X.columns if col in metadata.columns and col != 'subject_id']
    timeseries_features = [col for col in train_X.columns if col not in metadata_features and col != 'subject_id']

    print("\n" + "=" * 60)
    print("STAGE 1 — Data Quality")
    print("=" * 60)
    plot_missing_values(train_X[timeseries_features])
    plot_zero_variance(train_X, timeseries_features)
    analyze_zero_variance_features(train_X, timeseries_features)
    plot_class_balance(train_y)

    print("\n" + "=" * 60)
    print("STAGE 2 — Target Relationship")
    print("=" * 60)
    scores_df = compute_univariate_scores(train_X, train_y, timeseries_features, top_n=top_n)
    plot_univariate_scores(scores_df, top_n=top_n)
    plot_feature_distributions(train_X, train_y, features=scores_df.head(12)['feature'].tolist())

    subject_level = (
    train_X[['subject_id', 'ahi_stratum'] + metadata_features]
    .drop_duplicates('subject_id')
    .copy()
    )
    # subject_level['ahi_stratum'] = subject_level['ahi_stratum'].astype(str)

    # for feat in metadata_features:
    #     try:
    #         print(f"{feat}: dtype={subject_level[feat].dtype}, ndim={subject_level[feat].ndim}, sample={subject_level[feat].iloc[0]}")
    #     except Exception as e:
    #         print(f"{feat}: ERROR — {e}")

    # return None

    plot_metadata_by_stratum(train_X, metadata_features)

    print("\n" + "=" * 60)
    print("STAGE 3 — Feature Redundancy")
    print("=" * 60)
    top_ts_features = scores_df.head(50)['feature'].tolist()
    plot_correlation_matrix(train_X, timeseries_features, features=top_ts_features,
                            threshold=correlation_threshold)
    get_redundant_feature_groups(train_X, timeseries_features, threshold=correlation_threshold)

    if feature_groups:
        print("\n" + "=" * 60)
        print("STAGE 4 — Domain-informed Grouping")
        print("=" * 60)
        plot_group_importance(scores_df, feature_groups)

    return scores_df, metadata_features, timeseries_features
