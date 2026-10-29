import optuna

# 1. Connect to the database
study = optuna.load_study(
    study_name="rt_detr_coco5k_run", storage="sqlite:///optuna_coco5k.db"
)

# 2. Print best trial details
print("=" * 45)
print(f"Best Trial Number: {study.best_trial.number}")
print(f"Lowest Loss:       {study.best_value:.4f}")
print("Best Parameters:")
for param, val in study.best_params.items():
    print(f"  - {param}: {val}")
print("=" * 45)

# 3. View all trials as a Pandas DataFrame
df = study.trials_dataframe()
print(
    df[
        [
            "number",
            "value",
            "params_max_lr",
            "params_pct_start",
            "state",
            "duration",
        ]
    ]
)