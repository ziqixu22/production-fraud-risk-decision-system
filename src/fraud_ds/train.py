from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib, mlflow, pandas as pd, yaml
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV
from .data import load_and_merge, temporal_split
from .features import add_features, selected_columns
from .metrics import classification_metrics
from .policy import optimize_threshold


def build_model(numeric, categorical, params):
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20))
        ]), categorical)
    ])
    clf = LGBMClassifier(
        n_estimators=params.get("n_estimators", 500),
        learning_rate=params.get("learning_rate", 0.05),
        num_leaves=params.get("num_leaves", 31),
        random_state=params.get("random_state", 42),
        class_weight="balanced", n_jobs=-1)
    return Pipeline([("pre", pre), ("model", clf)])


def main(transaction, identity, artifact_dir, config_path):
    cfg = yaml.safe_load(Path(config_path).read_text())
    df = add_features(load_and_merge(transaction, identity))
    s = temporal_split(df, cfg["split"]["train_fraction"], cfg["split"]["validation_fraction"])
    numeric, categorical = selected_columns(s.train)
    features = numeric + categorical
    Xtr, ytr = s.train[features], s.train["isFraud"]
    Xv, yv = s.valid[features], s.valid["isFraud"]
    Xt, yt = s.test[features], s.test["isFraud"]
    base = build_model(numeric, categorical, cfg["model"])
    base.fit(Xtr, ytr)
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    calibrated.fit(Xv, yv)
    pv, pt = calibrated.predict_proba(Xv)[:,1], calibrated.predict_proba(Xt)[:,1]
    vm, tm = classification_metrics(yv, pv), classification_metrics(yt, pt)
    pcfg = cfg["policy"]
    policy = optimize_threshold(yv, pv, pcfg["fraud_miss_cost"], pcfg["false_decline_cost"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, artifact_dir / "model.joblib")
    meta = {"features": features, "threshold": policy.threshold,
            "review_threshold": policy.threshold * pcfg.get("review_threshold_multiplier", .6),
            "valid_metrics": vm, "test_metrics": tm, "policy": policy.__dict__}
    (artifact_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    mlflow.set_experiment("fraud-risk-decision")
    with mlflow.start_run():
        mlflow.log_params(cfg["model"])
        mlflow.log_metrics({f"valid_{k}": v for k,v in vm.items()})
        mlflow.log_metrics({f"test_{k}": v for k,v in tm.items()})
        mlflow.log_metric("decision_threshold", policy.threshold)
        mlflow.log_metric("validation_expected_cost", policy.expected_cost)
        mlflow.log_artifacts(str(artifact_dir), artifact_path="decision_system")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--transaction", type=Path, required=True)
    ap.add_argument("--identity", type=Path)
    ap.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    ap.add_argument("--config", default="configs/model.yaml")
    a = ap.parse_args()
    main(a.transaction, a.identity, a.artifact_dir, a.config)
