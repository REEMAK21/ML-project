import os, sys, json, hashlib, subprocess, platform, logging
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier

from .config import TrainCfg
from sklearn.cluster import KMeans
import logging
log = logging.getLogger(__name__)


def fit_kmeans(df: pd.DataFrame, *, k: int = 5):
    num = df.select_dtypes(include=["number"]).columns
    cat = df.select_dtypes(exclude=["number"]).columns

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
            ]), num),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore")),
            ]), cat),
        ]
    )

    model = Pipeline([
        ("pre", pre),
        ("kmeans", KMeans(n_clusters=k, n_init="auto", random_state=42)),
    ])

    model.fit(df)
    return model





def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()




def run_train(cfg: TrainCfg, *, root: Path, run_tag: str = "clf") -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_id = f"{ts}__{run_tag}__session{cfg.session_id}"
    run_dir = root / "models" / "runs" / run_id

    for d in ["metrics", "plots", "tables", "schema", "env", "model"]:
        (run_dir / d).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log.info("Run dir: %s", run_dir)

    df = pd.read_parquet(cfg.features_path)
    assert cfg.target in df.columns, f"Missing target: {cfg.target}"
    df = df.dropna(subset=[cfg.target]).reset_index(drop=True)

    # Optional: enforce time sorting for time-based evaluation
    if cfg.time_col:
        assert cfg.time_col in df.columns, f"Missing time_col: {cfg.time_col}"
        df = df.sort_values(cfg.time_col).reset_index(drop=True)

    # Build schema contract: required features vs optional IDs
    feature_cols = [c for c in df.columns if c not in {cfg.target, *cfg.id_cols}]
    schema = {
        "target": cfg.target,
        "required_feature_columns": feature_cols,
        "optional_id_columns": [c for c in cfg.id_cols if c in df.columns],
        "feature_dtypes": {c: str(df[c].dtype) for c in feature_cols},
        "datetime_columns": [c for c in feature_cols if "datetime" in str(df[c].dtype).lower()],
        "policy_unknown_categories": "tolerant (OneHotEncoder handle_unknown=ignore)",
        "forbidden_columns": [cfg.target],
    }
    (run_dir / "schema" / "input_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    # Environment capture
    (run_dir / "env" / "pip_freeze.txt").write_text(_pip_freeze(), encoding="utf-8")
    env_meta = {
        "python_version": sys.version,
        "python_version_short": platform.python_version(),
        "platform": platform.platform(),
    }
    (run_dir / "env" / "env_meta.json").write_text(json.dumps(env_meta, indent=2), encoding="utf-8")

    # Keep PyCaret outputs inside run_dir
    cwd = Path.cwd()
    os.chdir(run_dir)

    # ---- baseline dummy classifier ----
    from sklearn.model_selection import train_test_split
    X = df[feature_cols]
    y = df[cfg.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=cfg.train_size, random_state=cfg.session_id
    )

    # Basic model (since you said: no model changes)
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Metrics
    acc = (preds == y_test).mean()
    (Path("metrics") / "baseline_holdout.json").write_text(
        json.dumps({"accuracy": float(acc)}, indent=2)
    )

    # Save model
    import joblib
    joblib.dump(model, Path("model") / "model.joblib")

    # Update registry
    registry = root / "models" / "registry"
    registry.mkdir(parents=True, exist_ok=True)
    (registry / "latest.txt").write_text(run_id)

    os.chdir(cwd)  # back to original directory
    return run_dir

def _pip_freeze() -> str:
    try:
        return subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    except Exception as e:
        return f"# pip freeze failed: {e!r}\n"