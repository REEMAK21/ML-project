from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
import joblib, json, os
from datetime import datetime, timezone
from .config import TrainCfg


def run_train(cfg: TrainCfg, *, root: Path = Path(".")):
    # --- 1. Load data ---
    # df = pd.read_parquet(cfg.features_path)
    df = pd.read_csv(cfg.features_path)

    
    # --- 2. Separate X and y ---
    X = df.drop(columns=[cfg.target])
    y = df[cfg.target]

    # --- 3. Split train/holdout ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=cfg.train_size, random_state=cfg.session_id
    )

    # --- 4. Dummy baseline ---
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = float((preds == y_test).mean())

    # --- 5. run folder ---
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_id = f"{ts}__baseline__seed{cfg.session_id}"
    run_dir = root / "models" / "runs" / run_id
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "model").mkdir(parents=True, exist_ok=True)

    # --- 6. Save metrics ---
    (run_dir / "metrics" / "baseline_holdout.json").write_text(
        json.dumps({"accuracy": acc}, indent=2)
    )

    # --- 7. Save model ---
    joblib.dump(model, run_dir / "model" / "model.joblib")

    # --- 8. Update registry ---
    registry = root / "models" / "registry"
    registry.mkdir(parents=True, exist_ok=True)
    (registry / "latest.txt").write_text(run_id)

    print("✔ Training complete")
    print(f"📁 Run: {run_dir}")
    print(f"📌 Accuracy: {acc}")
    return run_dir
