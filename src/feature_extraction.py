"""
Feature extraction for DREAMT 2.1.0 dataset
- Extracts features from signals for non 'P' sleep stages:
        basic stats from time-domain signals, bandpowers (extended from -5 epochs to end of current epoch),
        HRV metrics, EDA phasic features, cross-signal features
- Includes target variables from each epoch:
    sleep stage:
        `sleep_stage` (mode), `sleep_stage_transition` (end of epoch, if different from mode),
        `sleep_stage_trans_prop` (proportion of transitional sleep stage in epoch),
    presence of apnea events (0 or 1):
        `apnea_obstructive`, `apnea_central`, `apnea_hypopnea`, `apnea_mixed`
- Output:
    Parquet file per subject with one row per epoch, containing extracted features and target variables
- Example usage:
    python feature_extraction.py --input_dir /path/to/csvs --output_dir /path/to/features
"""


import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from scipy.signal import welch, find_peaks
from scipy.stats import entropy

from config import dir_config
compiled_dir = dir_config.data.compiled
processed_dir = dir_config.data.processed


# Variable mapping (raw column name to standardized name)
vars_to_consider = {
    'timestamp': 'timestamp',
    'c4-m1': 'eeg_c4',
    'f4-m1': 'eeg_f4',
    'o2-m1': 'eeg_o2',
    'fp1-o2': 'eeg_fp1',
    't3-cz': 'eeg_t3',
    'cz-t4': 'eeg_cz',
    'e1': 'eog_e1',
    'e2': 'eog_e2',
    'chin': 'emg_chin',
    'lat': 'emg_lat',
    'rat': 'emg_rat',
    'ptaf': 'resp_ptaf',
    'flow': 'resp_flow',
    'thorax': 'resp_thorax',
    'abdomen': 'resp_abdomen',
    'bvp': 'bvp',
    'ibi': 'ibi',
    'eda': 'eda',
    'temp': 'temp',
    'hr': 'hr',
    'snore': 'snore',
    'sao2': 'sao2',
    'acc_x': 'acc_x',
    'acc_y': 'acc_y',
    'acc_z': 'acc_z',
    'sleep_stage': 'sleep_stage',
    'obstructive_apnea': 'apnea_obstructive',
    'central_apnea': 'apnea_central',
    'hypopnea': 'apnea_hypopnea',
    'multiple_events': 'apnea_mixed',
    }

# Epochs
EPOCH_SEC = 5
EXTENDED_EPOCH_SEC = 25  # for bandpower calculation

# EEG / EOG / EMG bands
EEG_BANDS = {"delta": (0.5,4), "theta":(4,8), "alpha":(8,12), "sigma":(12,15),
             "beta":(15,30),"gamma":(30,45),"high_gamma":(45,80)}
EOG_BANDS = {"slow":(0.1,1),"delta":(1,4),"theta":(4,8),"high":(8,15)}
EMG_BANDS = {"low":(0.5,10),"medium":(10,30),"high":(30,100)}

# -----------------------------
# Helper functions
# -----------------------------
def bandpower(data, fs, band):
    low, high = band
    freqs, psd = welch(data, fs=fs, nperseg=min(4*fs,len(data)))
    idx = np.logical_and(freqs>=low, freqs<=high)
    return np.trapz(psd[idx], freqs[idx])

def extract_basic_stats(signal):
    return {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'min': np.min(signal),
        'max': np.max(signal),
        'range': np.max(signal)-np.min(signal),
        'slope': 0 if np.std(signal) == 0 else np.polyfit(np.arange(len(signal)), signal,1)[0]
    }

def compute_acc_magnitude(acc_x,acc_y,acc_z):
    return np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

def compute_hrv_metrics(ibi_ms):
    """
    ibi_ms: IBI in milliseconds
    """
    if len(ibi_ms)<2:
        return {'sdnn': np.nan, 'rmssd': np.nan, 'pnn50': np.nan}
    diff = np.diff(ibi_ms)
    sdnn = np.std(ibi_ms)
    rmssd = np.sqrt(np.mean(diff**2))
    pnn50 = np.sum(np.abs(diff)>50)/len(diff)  # fraction >50ms
    return {'sdnn': sdnn, 'rmssd': rmssd, 'pnn50': pnn50}

def extract_eda_phasic(eda_signal, threshold=0.01):
    """Count phasic peaks by thresholding derivative"""
    if len(eda_signal)<2:
        return {'eda_scr_count':0,'eda_scr_mean_amp':0}
    diff = np.diff(eda_signal)
    peaks = diff > threshold
    count = np.sum(peaks)
    mean_amp = np.mean(diff[peaks]) if count>0 else 0
    return {'eda_scr_count': int(count), 'eda_scr_mean_amp': mean_amp}

def compute_respiratory_rate(resp_flow, fs, min_hz=0.1, max_hz=0.6):
    """Estimate respiratory rate from flow signal via peak detection.

    Returns breaths-per-minute and inter-breath interval CV (irregularity).
    """
    if len(resp_flow) < 2:
        return {'resp_rate_bpm': np.nan, 'resp_ibi_cv': np.nan}

    min_dist = int(fs / max_hz)  # minimum samples between peaks
    peaks, _ = find_peaks(resp_flow, distance=min_dist, height=np.percentile(resp_flow, 25))

    n_peaks = len(peaks)
    duration_sec = len(resp_flow) / fs
    bpm = (n_peaks / duration_sec) * 60 if duration_sec > 0 else np.nan

    if n_peaks >= 2:
        ibi = np.diff(peaks) / fs  # inter-breath intervals in seconds
        cv = np.std(ibi) / np.mean(ibi) if np.mean(ibi) > 0 else np.nan
    else:
        cv = np.nan

    return {'resp_rate_bpm': bpm, 'resp_ibi_cv': cv}


def compute_thorax_abdomen_coherence(thorax, abdomen):
    """Pearson correlation between thorax and abdomen effort signals.

    Positive = in-phase (normal breathing).
    Negative = paradoxical (hallmark of obstructive apnea).
    """
    if len(thorax) < 2 or np.std(thorax) == 0 or np.std(abdomen) == 0:
        return np.nan
    return float(np.corrcoef(thorax, abdomen)[0, 1])


def compute_sao2_desaturation_rate(sao2):
    """Most negative per-sample change in SpO2 — flags rapid desaturation."""
    if len(sao2) < 2:
        return {'sao2_desat_rate': np.nan}
    return {'sao2_desat_rate': float(np.diff(sao2.astype(float)).min())}


def compute_effort_flow_paradox(thorax, abdomen, resp_flow, effort_threshold=0.02, flow_threshold=0.02):
    """Flag: respiratory effort present (thorax/abdomen moving) but no airflow.

    Returns 1 if paradoxical breathing is detected, else 0.
    Classic sign of obstructive apnea.
    """
    effort_std = (np.std(thorax) + np.std(abdomen)) / 2
    flow_std = np.std(resp_flow)
    return int(effort_std > effort_threshold and flow_std < flow_threshold)


def compute_hrv_extended(ibi_ms, fs_ibi=100):
    """HRV metrics on a longer IBI window plus LF/HF frequency ratio.

    ibi_ms: IBI signal values (milliseconds at original sampling rate).
    """
    if len(ibi_ms) < 4:
        return {'ibi_sdnn_ext': np.nan, 'ibi_rmssd_ext': np.nan,
                'ibi_pnn50_ext': np.nan, 'ibi_lf_hf': np.nan}

    diff = np.diff(ibi_ms)
    sdnn  = float(np.std(ibi_ms))
    rmssd = float(np.sqrt(np.mean(diff ** 2)))
    pnn50 = float(np.sum(np.abs(diff) > 50) / len(diff))

    # LF/HF ratio via Welch PSD on the IBI series
    try:
        freqs, psd = welch(ibi_ms, fs=fs_ibi, nperseg=min(4 * fs_ibi, len(ibi_ms)))
        lf = float(np.trapz(psd[(freqs >= 0.04) & (freqs < 0.15)],
                             freqs[(freqs >= 0.04) & (freqs < 0.15)]))
        hf = float(np.trapz(psd[(freqs >= 0.15) & (freqs < 0.40)],
                             freqs[(freqs >= 0.15) & (freqs < 0.40)]))
        lf_hf = lf / hf if hf > 0 else np.nan
    except Exception:
        lf_hf = np.nan

    return {'ibi_sdnn_ext': sdnn, 'ibi_rmssd_ext': rmssd,
            'ibi_pnn50_ext': pnn50, 'ibi_lf_hf': lf_hf}


def compute_snore_bursts(snore_signal, threshold_pct=75):
    """Count snore events and measure longest burst duration above threshold.

    Returns snore event count and max burst length (in samples).
    """
    if len(snore_signal) < 2:
        return {'snore_event_count': 0, 'snore_max_burst': 0}

    threshold = np.percentile(np.abs(snore_signal), threshold_pct)
    above = (np.abs(snore_signal) > threshold).astype(int)

    # Count leading-edge transitions (0→1) as events
    events = int(np.sum(np.diff(np.concatenate([[0], above])) == 1))

    # Longest consecutive run above threshold
    max_burst = 0
    current = 0
    for v in above:
        if v:
            current += 1
            max_burst = max(max_burst, current)
        else:
            current = 0

    return {'snore_event_count': events, 'snore_max_burst': max_burst}


# -----------------------------
# Main extraction function
# -----------------------------
def extract_full_multimodal(df, signals, epoch_sec=EPOCH_SEC, fs=100):
    start_time = df['timestamp'].min()
    end_time = df['timestamp'].max()
    n_epochs = int((end_time-start_time)//epoch_sec)
    extended_time = EXTENDED_EPOCH_SEC

    ## Calculate acceleration magnitude
    if all(x in df.columns for x in ['acc_x','acc_y','acc_z']) and len(df)>0:
        df['acc_magnitude'] = compute_acc_magnitude(df['acc_x'].values,
                                                    df['acc_y'].values,
                                                    df['acc_z'].values)
    all_features = []

    for i in range(n_epochs):
        if (i+1) % (n_epochs//10) == 0:
            print(f"Extracted features : ({(i+1)/n_epochs:.0%})")
        epoch_start = start_time + i*epoch_sec
        epoch_end = epoch_start + epoch_sec
        epoch_df = df[(df['timestamp']>=epoch_start)&(df['timestamp']<epoch_end)]
        extended_epoch_df = df[(df['timestamp']>=epoch_start-extended_time)&(df['timestamp']<epoch_end)]
        epoch_feats = {'epoch_id':i, 'epoch_start':epoch_start, 'epoch_end':epoch_end}

        # -----------------------------
        # Time-domain features
        # -----------------------------
        for sig in signals:
            if sig in ['acc_x','acc_y','acc_z']:
                continue
            if sig in epoch_df.columns and len(epoch_df[sig])>0:
                epoch_feats.update({f'{sig}_{k}':v for k,v in extract_basic_stats(epoch_df[sig].values).items()})


        # -----------------------------
        # EEG bandpowers
        # -----------------------------
        for sig in signals:
            if sig.lower().startswith('eeg') and sig in epoch_df.columns:
                eeg_signal = extended_epoch_df[sig].values
                for band_name, band_range in EEG_BANDS.items():
                    epoch_feats[f'{sig}_bp_{band_name}'] = bandpower(eeg_signal, fs, band_range)

        # -----------------------------
        # EOG bandpowers
        # -----------------------------
        for sig in signals:
            if sig.lower().startswith('eog') and sig in epoch_df.columns:
                eog_signal = extended_epoch_df[sig].values
                for band_name, band_range in EOG_BANDS.items():
                    epoch_feats[f'{sig}_bp_{band_name}'] = bandpower(eog_signal, fs, band_range)

        # -----------------------------
        # EMG bandpowers
        # -----------------------------
        for sig in signals:
            if sig.lower().startswith('emg') and sig in epoch_df.columns:
                emg_signal = extended_epoch_df[sig].values
                for band_name, band_range in EMG_BANDS.items():
                    epoch_feats[f'{sig}_bp_{band_name}'] = bandpower(emg_signal, fs, band_range)

        # -----------------------------
        # IBI / HRV metrics (short window — kept for backward compatibility)
        # -----------------------------
        if 'ibi' in epoch_df.columns and len(epoch_df['ibi'])>1:
            ibi_ms = epoch_df['ibi'].values
            hrv = compute_hrv_metrics(ibi_ms)
            epoch_feats.update({f'ibi_{k}':v for k,v in hrv.items()})

        # HRV on extended 25s window + LF/HF ratio
        if 'ibi' in extended_epoch_df.columns and len(extended_epoch_df['ibi']) > 3:
            hrv_ext = compute_hrv_extended(extended_epoch_df['ibi'].values, fs_ibi=fs)
            epoch_feats.update(hrv_ext)

        # -----------------------------
        # EDA phasic peaks
        # -----------------------------
        if 'eda' in epoch_df.columns and len(epoch_df['eda'])>1:
            eda_feats = extract_eda_phasic(epoch_df['eda'].values)
            epoch_feats.update(eda_feats)

        # -----------------------------
        # Cross-signal features
        # -----------------------------
        corr_threshold = 1e-3
        # HR/BVP correlation
        if 'hr' in epoch_df.columns and 'bvp' in epoch_df.columns and len(epoch_df)>1:
            hr = epoch_df['hr'].values
            bvp = epoch_df['bvp'].values
            # check for constant signals to avoid NaN correlation

            if np.std(hr) > corr_threshold and np.std(bvp) > corr_threshold:
                epoch_feats['hr_bvp_corr'] = np.corrcoef(hr,bvp)[0,1]
            else:
                epoch_feats['hr_bvp_corr'] = 0

        # ACC + EDA arousal proxy
        if 'eda' in epoch_df.columns and 'acc_magnitude' in epoch_df.columns and len(epoch_df)>1:
            acc_mag = epoch_df['acc_magnitude'].values
            # correlation as simple arousal proxy
            if np.std(acc_mag)>corr_threshold and np.std(epoch_df['eda'].values)>corr_threshold:
                epoch_feats['acc_eda_corr'] = np.corrcoef(acc_mag, epoch_df['eda'].values)[0,1]
            else:
                epoch_feats['acc_eda_corr'] = 0

        # -----------------------------
        # Respiratory clinical features
        # -----------------------------
        # Respiratory rate (breaths per minute) and irregularity — use extended window for stability
        if 'resp_flow' in extended_epoch_df.columns and len(extended_epoch_df['resp_flow']) > fs:
            rr = compute_respiratory_rate(extended_epoch_df['resp_flow'].values, fs=fs)
            epoch_feats.update(rr)

        # Thorax-abdomen phase coherence (negative = paradoxical breathing = obstructive sign)
        if ('resp_thorax' in extended_epoch_df.columns and
                'resp_abdomen' in extended_epoch_df.columns and
                len(extended_epoch_df) > 1):
            epoch_feats['resp_thorax_abdomen_coherence'] = compute_thorax_abdomen_coherence(
                extended_epoch_df['resp_thorax'].values,
                extended_epoch_df['resp_abdomen'].values,
            )

        # Effort-flow paradox flag (effort present but no airflow)
        if all(c in epoch_df.columns for c in ['resp_thorax', 'resp_abdomen', 'resp_flow']):
            epoch_feats['resp_effort_flow_paradox'] = compute_effort_flow_paradox(
                epoch_df['resp_thorax'].values,
                epoch_df['resp_abdomen'].values,
                epoch_df['resp_flow'].values,
            )

        # SpO2 desaturation rate (most negative per-sample drop)
        if 'sao2' in epoch_df.columns and len(epoch_df['sao2']) > 1:
            epoch_feats.update(compute_sao2_desaturation_rate(epoch_df['sao2'].values))

        # -----------------------------
        # Snore burst features
        # -----------------------------
        if 'snore' in epoch_df.columns and len(epoch_df['snore']) > 1:
            epoch_feats.update(compute_snore_bursts(epoch_df['snore'].values))

        # -----------------------------
        # EEG arousal ratio: (beta+gamma) / (delta+theta) — arousals from apnea increase this
        # -----------------------------
        for sig in signals:
            if sig.lower().startswith('eeg') and sig in epoch_df.columns:
                eeg_ext = extended_epoch_df[sig].values
                bp_delta = bandpower(eeg_ext, fs, EEG_BANDS['delta'])
                bp_theta = bandpower(eeg_ext, fs, EEG_BANDS['theta'])
                bp_beta  = bandpower(eeg_ext, fs, EEG_BANDS['beta'])
                bp_gamma = bandpower(eeg_ext, fs, EEG_BANDS['gamma'])
                denom = bp_delta + bp_theta
                epoch_feats[f'{sig}_arousal_ratio'] = (bp_beta + bp_gamma) / denom if denom > 0 else np.nan

        # ------------------------------
        # Target variables
        # ------------------------------
        # Sleep stage (mode in epoch)
        if 'sleep_stage' in epoch_df.columns and len(epoch_df['sleep_stage'])>0:
            s0 = epoch_df['sleep_stage'].iloc[0]     # proxy for mode, based on observation
            s1 = epoch_df['sleep_stage'].iloc[-1]    # end stage
            epoch_feats['sleep_stage'] = s0
            # Sleep stage transition (last vs mode), proportion of epoch in transition
            if s1 == s0:
                epoch_feats['sleep_stage_transition'] = pd.NA
                epoch_feats['sleep_stage_trans_prop'] = 0.0
            else:
                epoch_feats['sleep_stage_transition'] = s1
                epoch_feats['sleep_stage_trans_prop'] = float((epoch_df['sleep_stage'] == s1).mean())

        # Apnea events (present in epoch)
        for apnea_type in ['apnea_obstructive', 'apnea_central', 'apnea_hypopnea', 'apnea_mixed']:
            if apnea_type in epoch_df.columns:
                # 1 if any event in epoch, else 0
                epoch_feats[apnea_type] = epoch_df[apnea_type].max()

        all_features.append(epoch_feats)

    return pd.DataFrame(all_features)


def process_one_subject(csv_path, out_dir, epoch_sec=EPOCH_SEC, fs=100):
    csv_path = Path(csv_path)
    sid = csv_path.stem.split("_")[0]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sid}.parquet"

    # skip if already computed
    if out_path.exists():
        return sid, "skipped"

    print(f"Processing {sid}...")

    df_original = pd.read_csv(csv_path)

    # normalize column names
    df_original.columns = df_original.columns.str.lower().str.replace(' ','')
    # remove epochs with 'P' sleep stage (if exists)
    df_original = df_original[df_original.sleep_stage != 'P'].reset_index(drop=True)
    df = pd.DataFrame()
    for old_var, new_var in vars_to_consider.items():
        if old_var in df_original.columns:
            df[new_var] = df_original[old_var]

    # replace NA with 0, missingness indicator
    null_columns = []
    for column, count in df.isnull().sum().items():
        if count > 0:
            #print(f"Column: {column}, Null Count: {count}")
            null_columns.append(column)
    for col in null_columns:
        df[col] = df[col].apply(lambda x: 0 if pd.isna(x) else 1)

    signals = list(vars_to_consider.values())

    # Remove timestamp and label/indicator columns (not continuous signals)
    exclude_from_features = {'timestamp', 'sleep_stage', 'apnea_obstructive',
                             'apnea_central', 'apnea_hypopnea', 'apnea_mixed'}
    signals = [s for s in signals if s not in exclude_from_features]
    df['sleep_stage'] = df['sleep_stage'].astype('category')
    feats = extract_full_multimodal(df, signals=signals, epoch_sec=epoch_sec, fs=fs)

    feats.to_parquet(out_path, index=False)  # snappy by default in many installs

    return sid, "done"

def run_all(input_dir, out_dir, epoch_sec, fs, max_workers=None):
    input_dir = Path(input_dir)
    csv_files = sorted(input_dir.glob("*.csv"))

    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 6) - 1)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(process_one_subject, str(p), str(out_dir), epoch_sec, fs)
            for p in csv_files
        ]

        for fut in as_completed(futures):
            sid, status = fut.result()
            print(sid, status)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch_sec", type=int, default=EPOCH_SEC)
    parser.add_argument("--fs", type=int, default=100)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--input_dir", type=str, default=str(Path(compiled_dir, "data_100Hz")))
    parser.add_argument("--output_dir", type=str, default=str(Path(processed_dir)))

    args = parser.parse_args()

    # Start with max_workers=4 or 6 and scale up
    run_all(
        input_dir=args.input_dir,
        out_dir=args.output_dir,
        epoch_sec=args.epoch_sec,
        fs=args.fs,
        max_workers=args.max_workers,
    )
