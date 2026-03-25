"""
DREAMT Dataset — Apnea Prediction Feature Tiers
================================================
Features are tiered by predictive power for apnea event prediction
(1–2 epochs ahead, ~30–60s horizon), across all apnea types.

Tier structure:
  TIER_1  Must Have      — Direct physiological precursors; strongest evidence
  TIER_2  Good to Have   — Strong secondary signals; well-evidenced
  TIER_3  Okay to Have   — Useful context or indirect signal
  TIER_4  Maybe          — Indirect, redundant, or artifact-prone; test empirically
  TIER_5  Not Useful     — Wrong timescale or no physiological link at epoch level

Convenience aggregates:
  ALL_TIERS              — Flat list of all features, ordered T1→T5
  RECOMMENDED            — T1 + T2 (good starting point for modelling)
  FEATURES_BY_MODALITY   — Dict keyed by modality name → list of features
  TIER_MAP               — Dict mapping every feature → its tier integer (1–5)
"""

# ---------------------------------------------------------------------------
# TIER 1 — Must Have
# Direct physiological precursors to apnea events
# ---------------------------------------------------------------------------

# Respiratory — Airflow (PTAF + Flow)
# Airflow reduction/cessation is the definition of apnea/hypopnea.
# All stats are informative: mean/std capture epoch dynamics, slope captures
# onset trajectory, min/range capture amplitude collapse.
RESP_AIRFLOW_T1 = [
    "resp_ptaf_mean", "resp_ptaf_std", "resp_ptaf_min",
    "resp_ptaf_max", "resp_ptaf_range", "resp_ptaf_slope",
    "resp_flow_mean", "resp_flow_std", "resp_flow_min",
    "resp_flow_max", "resp_flow_range", "resp_flow_slope",
]

# Respiratory — Effort (Thorax + Abdomen)
# Distinguishes obstructive vs. central and detects paradoxical breathing onset.
# Slope captures crescendo effort that directly precedes obstruction.
RESP_EFFORT_T1 = [
    "resp_thorax_mean", "resp_thorax_std", "resp_thorax_min",
    "resp_thorax_max", "resp_thorax_range", "resp_thorax_slope",
    "resp_abdomen_mean", "resp_abdomen_std", "resp_abdomen_min",
    "resp_abdomen_max", "resp_abdomen_range", "resp_abdomen_slope",
]

# Respiratory — Derived cross-signal features
# Encode obstructive physiology that no single-channel stat can capture.
RESP_DERIVED_T1 = [
    "resp_rate_bpm",
    "resp_ibi_cv",
    "resp_thorax_abdomen_coherence",
    "resp_effort_flow_paradox",
]

# SpO2 — All stats
# Primary outcome of apnea (oxygen desaturation) and strongest cyclic
# recurrence signal. All stats matter across distributional shape.
SPO2_T1 = [
    "sao2_mean", "sao2_std", "sao2_min",
    "sao2_max", "sao2_range", "sao2_slope",
    "sao2_desat_rate",
]

# IBI / HRV — All stats and metrics
# Cyclic bradycardia-tachycardia is one of the most validated cardiac
# signatures in sleep medicine. min/max capture the tachycardic peak and
# bradycardic nadir; RMSSD/pNN50 capture parasympathetic withdrawal.
IBI_HRV_T1 = [
    "ibi_mean", "ibi_std", "ibi_min", "ibi_max", "ibi_range", "ibi_slope",
    "ibi_sdnn", "ibi_rmssd", "ibi_pnn50",
    "ibi_sdnn_ext", "ibi_rmssd_ext", "ibi_pnn50_ext",
    "ibi_lf_hf",
]

TIER_1 = (
    RESP_AIRFLOW_T1
    + RESP_EFFORT_T1
    + RESP_DERIVED_T1
    + SPO2_T1
    + IBI_HRV_T1
)

# ---------------------------------------------------------------------------
# TIER 2 — Good to Have
# Strong secondary signals; well-evidenced but one step removed
# ---------------------------------------------------------------------------

# HR — Carries the same autonomic apnea signature as IBI; may be computed
# on a different window so worth keeping alongside IBI.
HR_T2 = [
    "hr_mean", "hr_std", "hr_slope",
    "hr_min", "hr_max", "hr_range",
]

# BVP — Pulse wave amplitude drops 30–50% during apnea due to sympathetic
# vasoconstriction. Std captures PAT-surrogate pulse amplitude variability.
BVP_T2 = [
    "bvp_mean", "bvp_std", "bvp_slope",
    "bvp_min", "bvp_max", "bvp_range",
    "hr_bvp_corr",
]

# Snore — Directly precedes most OSA events; cessation marks apnea onset.
# Slope captures crescendo trajectory; event_count and max_burst encode
# temporal clustering.
SNORE_T2 = [
    "snore_mean", "snore_std", "snore_min",
    "snore_max", "snore_range", "snore_slope",
    "snore_event_count", "snore_max_burst",
]

# EEG — Central & Frontal channels (C4, Cz, F4)
# Sleep-regulatory cortex. Delta/theta/alpha/sigma shifts reflect subcortical
# arousal state changes that precede frank apnea. Arousal ratio is a direct
# index of cortical arousal burden.
EEG_CENTRAL_FRONTAL_BP_T2 = [
    "eeg_c4_bp_delta", "eeg_c4_bp_theta", "eeg_c4_bp_alpha",
    "eeg_c4_bp_sigma", "eeg_c4_arousal_ratio",
    "eeg_cz_bp_delta", "eeg_cz_bp_theta", "eeg_cz_bp_alpha",
    "eeg_cz_bp_sigma", "eeg_cz_arousal_ratio",
    "eeg_f4_bp_delta", "eeg_f4_bp_theta", "eeg_f4_bp_alpha",
    "eeg_f4_bp_sigma", "eeg_f4_arousal_ratio",
]

# EEG — C4, Cz, F4 time-domain stats
# Complement bandpower; capture slow amplitude drifts from deepening hypoxia
# or arousal build-up.
EEG_CENTRAL_FRONTAL_TD_T2 = [
    "eeg_c4_mean", "eeg_c4_std", "eeg_c4_slope",
    "eeg_cz_mean", "eeg_cz_std", "eeg_cz_slope",
    "eeg_f4_mean", "eeg_f4_std", "eeg_f4_slope",
]

# Demographics
DEMOGRAPHICS_T2 = [
    "bmi", "age", "gender_m",
]

TIER_2 = (
    HR_T2
    + BVP_T2
    + SNORE_T2
    + EEG_CENTRAL_FRONTAL_BP_T2
    + EEG_CENTRAL_FRONTAL_TD_T2
    + DEMOGRAPHICS_T2
)

# ---------------------------------------------------------------------------
# TIER_3 — Okay to Have
# Useful context or indirect signal; worth including but lower priority
# ---------------------------------------------------------------------------

# EEG — Occipital & Temporal channels (O2, Fp1, T3)
# Farther from sleep-regulatory anatomy. Delta/theta shifts are present but
# with weaker spatial specificity than central/frontal channels.
EEG_PERIPHERAL_BP_T3 = [
    "eeg_o2_bp_delta", "eeg_o2_bp_theta", "eeg_o2_bp_alpha",
    "eeg_o2_bp_sigma", "eeg_o2_arousal_ratio",
    "eeg_fp1_bp_delta", "eeg_fp1_bp_theta", "eeg_fp1_bp_alpha",
    "eeg_fp1_bp_sigma", "eeg_fp1_arousal_ratio",
    "eeg_t3_bp_delta", "eeg_t3_bp_theta", "eeg_t3_bp_alpha",
    "eeg_t3_bp_sigma", "eeg_t3_arousal_ratio",
]

# EEG — O2, Fp1, T3 time-domain stats
EEG_PERIPHERAL_TD_T3 = [
    "eeg_o2_mean", "eeg_o2_std", "eeg_o2_slope",
    "eeg_fp1_mean", "eeg_fp1_std", "eeg_fp1_slope",
    "eeg_t3_mean", "eeg_t3_std", "eeg_t3_slope",
]

# EOG — Primarily a sleep-stage marker (slow waves = NREM, rapid = REM).
# Slow/delta bands add stage context beyond the discrete sleep_stage label.
EOG_T3 = [
    "eog_e1_mean", "eog_e1_std", "eog_e1_slope",
    "eog_e1_bp_slow", "eog_e1_bp_delta",
    "eog_e2_mean", "eog_e2_std", "eog_e2_slope",
    "eog_e2_bp_slow", "eog_e2_bp_delta",
]

# EMG Chin — Submental tone mechanistically linked to upper airway patency.
# Genioglossus activity decreases in REM, contributing to OSA.
# Low/medium bands capture tonic activity.
EMG_CHIN_T3 = [
    "emg_chin_mean", "emg_chin_std", "emg_chin_slope",
    "emg_chin_bp_low", "emg_chin_bp_medium",
]

# EDA — Sympathetic arousal elevated peri-apnea. Phasic SCR features are
# more temporally specific; too slow for single-epoch precision but informative
# across epoch sequences.
EDA_T3 = [
    "eda_mean", "eda_std", "eda_slope",
    "eda_scr_count", "eda_scr_mean_amp",
]

# Cross-signal
CROSS_SIGNAL_T3 = [
    "acc_eda_corr",
]

# Clinical — Directly OSA-relevant comorbidities
# Static patient-level features; useful for stratification, not epoch-level.
CLINICAL_OSA_T3 = [
    "sd_osa", "sd_osa_snoring", "sd_snoring",
    "sd_eds", "sd_dyspnea",
    "med_osa", "med_hypertension", "med_diabetes",
]

TIER_3 = (
    EEG_PERIPHERAL_BP_T3
    + EEG_PERIPHERAL_TD_T3
    + EOG_T3
    + EMG_CHIN_T3
    + EDA_T3
    + CROSS_SIGNAL_T3
    + CLINICAL_OSA_T3
)

# ---------------------------------------------------------------------------
# TIER 4 — Maybe
# Indirect, redundant, or artifact-prone; test empirically
# ---------------------------------------------------------------------------

# EEG — All channels, high-freq bands (beta, gamma, high-gamma)
# No established role in apnea prediction. Heavy muscle artifact contamination,
# especially at temporal/frontal sites.
EEG_HIGH_FREQ_T4 = [
    "eeg_c4_bp_beta", "eeg_c4_bp_gamma", "eeg_c4_bp_high_gamma",
    "eeg_cz_bp_beta", "eeg_cz_bp_gamma", "eeg_cz_bp_high_gamma",
    "eeg_f4_bp_beta", "eeg_f4_bp_gamma", "eeg_f4_bp_high_gamma",
    "eeg_o2_bp_beta", "eeg_o2_bp_gamma", "eeg_o2_bp_high_gamma",
    "eeg_fp1_bp_beta", "eeg_fp1_bp_gamma", "eeg_fp1_bp_high_gamma",
    "eeg_t3_bp_beta", "eeg_t3_bp_gamma", "eeg_t3_bp_high_gamma",
]

# EEG — All channels, min/max/range
# Amplitude extremes dominated by artifact; signal already better captured
# by bandpower and mean/std.
EEG_EXTREMES_T4 = [
    "eeg_c4_min", "eeg_c4_max", "eeg_c4_range",
    "eeg_cz_min", "eeg_cz_max", "eeg_cz_range",
    "eeg_f4_min", "eeg_f4_max", "eeg_f4_range",
    "eeg_o2_min", "eeg_o2_max", "eeg_o2_range",
    "eeg_fp1_min", "eeg_fp1_max", "eeg_fp1_range",
    "eeg_t3_min", "eeg_t3_max", "eeg_t3_range",
]

# EOG — Higher bands + amplitude extremes
# No established sleep-stage or apnea relevance above delta.
EOG_T4 = [
    "eog_e1_bp_theta", "eog_e1_bp_high",
    "eog_e1_min", "eog_e1_max", "eog_e1_range",
    "eog_e2_bp_theta", "eog_e2_bp_high",
    "eog_e2_min", "eog_e2_max", "eog_e2_range",
]

# EMG Chin — High band and amplitude extremes
EMG_CHIN_T4 = [
    "emg_chin_bp_high",
    "emg_chin_min", "emg_chin_max", "emg_chin_range",
]

# EMG LAT/RAT — All stats and bands
# Designed for PLM detection, not apnea. Weak arousal signal at best.
EMG_LEG_T4 = [
    "emg_lat_mean", "emg_lat_std", "emg_lat_slope",
    "emg_lat_min", "emg_lat_max", "emg_lat_range",
    "emg_lat_bp_low", "emg_lat_bp_medium", "emg_lat_bp_high",
    "emg_rat_mean", "emg_rat_std", "emg_rat_slope",
    "emg_rat_min", "emg_rat_max", "emg_rat_range",
    "emg_rat_bp_low", "emg_rat_bp_medium", "emg_rat_bp_high",
]

# EDA — Amplitude extremes (add little over mean/std for slow tonic signal)
EDA_T4 = [
    "eda_min", "eda_max", "eda_range",
]

# Clinical — Indirect comorbidities and medications
# Unlikely to be informative at epoch level; marginal even as patient priors.
CLINICAL_INDIRECT_T4 = [
    "sd_insomnia", "sd_rls", "sd_bruxism", "sd_rbd",
    "sd_migraine", "sd_fatigue", "sd_hypersomnia", "sd_mci_and_sleep_apnea",
    "med_anxiety", "med_depression", "med_asthma", "med_body_pain",
    "med_arrhythmia", "med_gerd", "med_cad", "med_migraine", "med_dyspnea",
]

TIER_4 = (
    EEG_HIGH_FREQ_T4
    + EEG_EXTREMES_T4
    + EOG_T4
    + EMG_CHIN_T4
    + EMG_LEG_T4
    + EDA_T4
    + CLINICAL_INDIRECT_T4
)

# ---------------------------------------------------------------------------
# TIER 5 — Not Useful
# Wrong timescale or no physiological link at epoch level
# ---------------------------------------------------------------------------

# Temperature — Skin temperature dynamics operate on a timescale of minutes
# to hours; an order of magnitude too slow for 30s epoch-level apnea signal.
TEMP_T5 = [
    "temp_mean", "temp_std", "temp_min",
    "temp_max", "temp_range", "temp_slope",
]

TIER_5 = TEMP_T5

# ---------------------------------------------------------------------------
# Convenience aggregates
# ---------------------------------------------------------------------------

ALL_TIERS = TIER_1 + TIER_2 + TIER_3 + TIER_4 + TIER_5

# T1 + T2: solid baseline for modelling
RECOMMENDED = TIER_1 + TIER_2

# Flat lookup: feature name → tier integer
TIER_MAP: dict[str, int] = (
    {f: 1 for f in TIER_1}
    | {f: 2 for f in TIER_2}
    | {f: 3 for f in TIER_3}
    | {f: 4 for f in TIER_4}
    | {f: 5 for f in TIER_5}
)

# Features grouped by modality
FEATURES_BY_MODALITY: dict[str, list[str]] = {
    # Tier 1
    "resp_airflow":             RESP_AIRFLOW_T1,
    "resp_effort":              RESP_EFFORT_T1,
    "resp_derived":             RESP_DERIVED_T1,
    "spo2":                     SPO2_T1,
    "ibi_hrv":                  IBI_HRV_T1,
    # Tier 2
    "hr":                       HR_T2,
    "bvp":                      BVP_T2,
    "snore":                    SNORE_T2,
    "eeg_central_frontal_bp":   EEG_CENTRAL_FRONTAL_BP_T2,
    "eeg_central_frontal_td":   EEG_CENTRAL_FRONTAL_TD_T2,
    "demographics": DEMOGRAPHICS_T2,
    # Tier 3
    "eeg_peripheral_bp":        EEG_PERIPHERAL_BP_T3,
    "eeg_peripheral_td":        EEG_PERIPHERAL_TD_T3,
    "eog":                      EOG_T3,
    "emg_chin":                 EMG_CHIN_T3,
    "eda":                      EDA_T3,
    "cross_signal":             CROSS_SIGNAL_T3,
    "clinical_osa":             CLINICAL_OSA_T3,
    # Tier 4
    "eeg_high_freq":            EEG_HIGH_FREQ_T4,
    "eeg_extremes":             EEG_EXTREMES_T4,
    "eog_low_info":             EOG_T4,
    "emg_chin_low_info":        EMG_CHIN_T4,
    "emg_leg":                  EMG_LEG_T4,
    "eda_extremes":             EDA_T4,
    "clinical_indirect":        CLINICAL_INDIRECT_T4,
    # Tier 5
    "temperature":              TEMP_T5,
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_features(max_tier: int = 2) -> list[str]:
    """
    Return all features up to and including `max_tier`.

    Examples
    --------
    >>> get_features(1)        # Tier 1 only (must-have)
    >>> get_features(2)        # T1 + T2  (recommended baseline)
    >>> get_features(3)        # T1–T3
    """
    if not 1 <= max_tier <= 5:
        raise ValueError(f"max_tier must be 1–5, got {max_tier}")
    tiers = [TIER_1, TIER_2, TIER_3, TIER_4, TIER_5]
    result = []
    for t in tiers[:max_tier]:
        result.extend(t)
    return result


def get_tier(feature: str) -> int:
    """Return the tier (1–5) for a given feature name."""
    if feature not in TIER_MAP:
        raise KeyError(f"Feature '{feature}' not found in tier map.")
    return TIER_MAP[feature]


def filter_df(df, max_tier: int = 2):
    """
    Filter a DataFrame to columns in tiers 1..max_tier.
    Columns not present in ANY tier (e.g. epoch_id, epoch_start) are kept as-is.

    Parameters
    ----------
    df       : pandas DataFrame with DREAMT feature columns
    max_tier : include features up to this tier (default 2 = T1+T2)

    Returns
    -------
    DataFrame with only the selected feature columns (+ unrecognised columns).
    """
    selected = set(get_features(max_tier))
    # Keep columns that are either in the selected set OR not in the tier map at all
    keep = [c for c in df.columns if c in selected or c not in TIER_MAP]
    return df[keep]


# ---------------------------------------------------------------------------
# Quick self-test / summary (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tier_lists = [TIER_1, TIER_2, TIER_3, TIER_4, TIER_5]
    labels = [
        "Tier 1 — Must Have",
        "Tier 2 — Good to Have",
        "Tier 3 — Okay to Have",
        "Tier 4 — Maybe",
        "Tier 5 — Not Useful",
    ]
    print("DREAMT Feature Tier Summary")
    print("=" * 40)
    total = 0
    for label, tier in zip(labels, tier_lists):
        print(f"  {label}: {len(tier)} features")
        total += len(tier)
    print(f"  {'Total':.<30} {total} features")
    print()
    print(f"RECOMMENDED (T1+T2): {len(RECOMMENDED)} features")
    print()

    # Sanity check: no feature appears in more than one tier
    all_features = ALL_TIERS
    assert len(all_features) == len(set(all_features)), \
        "Duplicate feature detected across tiers!"
    print("Sanity check passed: no duplicate features across tiers.")
