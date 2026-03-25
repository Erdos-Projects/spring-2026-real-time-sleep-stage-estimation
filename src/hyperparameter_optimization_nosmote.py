"""
Fast LightGBM hyperparameter optimization with Optuna.

The key parallelism fix vs the notebook version:
  - Optuna runs trials sequentially (n_jobs=1)
  - Each LGBMClassifier uses ALL CPU cores (n_jobs=-1)
  Doing both in parallel causes thread contention and is slower overall.

Usage (from notebook):
    from src.hyperparameter_optimization import lgb_hyperparameter_fitting

    model, cv_aucpr, study = lgb_hyperparameter_fitting(
        train_X, train_y, fold_indices,
        n_trials=50,
        use_smote=False,
        random_seed=random_seed,
    )
"""

from __future__ import annotations

import warnings
import numpy as np
import optuna
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Main optimization function
# ---------------------------------------------------------------------------

def lgb_hyperparameter_fitting(
    train_X,
    train_y,
    fold_indices,
    n_trials: int = 50,
    random_seed: int = 42,
) -> tuple:
    """
    Hyperparameter search for LightGBM via Optuna.

    Parallelism strategy:
      - Optuna: n_jobs=1 (sequential trials)
      - LGBMClassifier: n_jobs=-1 (all CPU threads per trial)
      Running both in parallel causes CPU thrashing and is slower.

    Parameters
    ----------
    train_X      : DataFrame – full training features
    train_y      : Series   – full training labels
    fold_indices : list of (tr_idx, val_idx) arrays – CV folds
    n_trials     : number of Optuna trials
    use_smote    : apply SMOTE per fold; if False uses log-scaled scale_pos_weight
    random_seed  : RNG seed

    Returns
    -------
    (final_model, best_cv_aucpr, study)
    """

    neg = int((train_y == 0).sum())
    pos = int((train_y == 1).sum())
    scale_pos_weight = float(np.log1p(neg / pos))
    print(f"scale_pos_weight (log-scaled): {scale_pos_weight:.2f}")

    BASE_PARAMS: dict = {
        "scale_pos_weight": scale_pos_weight,
        "random_state":     random_seed,
        "metric":           "average_precision",
        "verbosity":        -1,
        # Each trial owns all CPU threads; Optuna runs trials sequentially.
        "n_jobs":           1,
    }

    def objective(trial: optuna.Trial) -> float:
        params = BASE_PARAMS | {
            "n_estimators":      trial.suggest_int("n_estimators", 1000, 10000, log=True),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            "max_depth":         trial.suggest_int("max_depth", 2, 4),
            "num_leaves":        trial.suggest_int("num_leaves", 10, 31),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 150),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.2, 0.6),
            "subsample":         trial.suggest_float("subsample", 0.3, 0.65),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0.5, 5.0),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1.0, 8.0),
        }

        fold_scores       = []
        fold_train_scores = []
        best_iterations   = []

        for tr_idx, val_idx in fold_indices:
            X_tr  = train_X.iloc[tr_idx]
            y_tr  = train_y.iloc[tr_idx]
            X_val = train_X.iloc[val_idx]
            y_val = train_y.iloc[val_idx]

            model = LGBMClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(50, first_metric_only=True, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )
            best_iterations.append(model.best_iteration_)

            fold_train_scores.append(
                average_precision_score(y_tr, model.predict_proba(X_tr)[:, 1])
            )
            fold_scores.append(
                average_precision_score(y_val, model.predict_proba(X_val)[:, 1])
            )

        mean_train = float(np.mean(fold_train_scores))
        mean_val   = float(np.mean(fold_scores))
        gap        = mean_train - mean_val

        trial.set_user_attr("mean_best_iteration", int(np.mean(best_iterations)))
        trial.set_user_attr("overfitting_gap", round(gap, 4))

        # if gap > 0.25:
        #     raise optuna.exceptions.TrialPruned()

        return mean_val

    # Sequential Optuna trials so each LGB model gets all CPU cores uncontested.
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_seed),
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=-1,
        show_progress_bar=True,
    )

    best_trial = study.best_trial
    print(f"\nBest CV AUC-PR:      {study.best_value:.4f}")
    print(f"Best params:          {best_trial.params}")
    print(f"Mean best iteration:  {best_trial.user_attrs['mean_best_iteration']}")
    print(f"Overfitting gap:      {best_trial.user_attrs['overfitting_gap']:.4f}")

    # Retrain on full training set at averaged best iteration — no early stopping.
    best_params = BASE_PARAMS | best_trial.params
    best_params["n_estimators"] = best_trial.user_attrs["mean_best_iteration"]
    best_params.pop("metric", None)

    final_model = LGBMClassifier(**best_params)

    final_model.fit(train_X, train_y)

    return final_model, study.best_value, study
