"""训练随机森林 + XGBoost 二分类模型并对比。

流程：加载记录 -> 特征提取（手写特征 + 可选 TF-IDF）-> 5 折交叉验证
      + 划分训练/测试集 -> 训练两模型 -> 指标对比 -> 保存最优模型。

手写特征只提取一次，交叉验证每折只重拟合 TF-IDF 词表（不重复提取），
既避免数据泄漏、也避免长邮件上重复提取的耗时。
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

import config
from data_loader import load_records
from features import email_text, extract_features, stack_features

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[警告] 未安装 xgboost，本次只训练随机森林。可执行: pip install xgboost")


def _stack(X_h: pd.DataFrame, texts: list[str], vectorizer=None, use_tfidf: bool = True, max_features: int = 5000):
    """拼接手写特征（已算好）与 TF-IDF。返回 (X, feature_names, vectorizer)。"""
    if use_tfidf:
        X, names, vec = stack_features(X_h, texts, vectorizer, max_features)
        return X, names, vec
    return X_h.values, list(X_h.columns), None


def _test_metrics(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "auc": roc_auc_score(y_test, proba),
    }


def _cv_scores(X_h: pd.DataFrame, texts: list[str], y: np.ndarray, model_factory, seed: int, use_tfidf: bool, max_features: int):
    """5 折交叉验证；每折在训练折上重新拟合 vectorizer（复用已提取的手写特征）。"""
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    accs, f1s = [], []
    for tr, te in skf.split(X_h, y):
        X_tr, _, vec = _stack(X_h.iloc[tr], [texts[i] for i in tr], None, use_tfidf, max_features)
        X_te, _, _ = _stack(X_h.iloc[te], [texts[i] for i in te], vec, use_tfidf, max_features)
        m = model_factory()
        m.fit(X_tr, y[tr])
        pred = m.predict(X_te)
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred))
    return float(np.mean(accs)), float(np.mean(f1s))


def _rf_factory(n_estimators: int, seed: int):
    return lambda: RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)


def _xgb_factory(n_estimators: int, seed: int):
    return lambda: XGBClassifier(n_estimators=n_estimators, max_depth=6, learning_rate=0.1, random_state=seed)


def run(
    records: list[dict],
    seed: int = config.SEED,
    test_size: float = 0.25,
    n_estimators: int = 200,
    use_tfidf: bool = True,
    max_features: int = 5000,
    plots: bool = True,
) -> dict:
    n_phish = sum(r["label"] for r in records)
    mode = "手写特征 + TF-IDF" if use_tfidf else "仅手写特征"
    print(f"\n数据集: {len(records)} 条（钓鱼 {n_phish} / 正常 {len(records) - n_phish}），特征模式：{mode}")

    # 手写特征只提取一次
    X_h = pd.DataFrame([extract_features(r) for r in records])
    texts = [email_text(r) for r in records]
    y = np.array([r["label"] for r in records])

    idx = np.arange(len(records))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)

    X_train, feature_names, vectorizer = _stack(
        X_h.iloc[train_idx], [texts[i] for i in train_idx], None, use_tfidf, max_features
    )
    X_test, _, _ = _stack(X_h.iloc[test_idx], [texts[i] for i in test_idx], vectorizer, use_tfidf, max_features)
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"特征维度: {X_train.shape[1]}（手写 {X_h.shape[1]} + TF-IDF {X_train.shape[1] - X_h.shape[1] if use_tfidf else 0}）")

    # 随机森林
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_cv_acc, rf_cv_f1 = _cv_scores(X_h, texts, y, _rf_factory(n_estimators, seed), seed, use_tfidf, max_features)
    rows = [{"model": "RandomForest", "cv_acc": rf_cv_acc, "cv_f1": rf_cv_f1, **_test_metrics(rf, X_test, y_test)}]

    # XGBoost
    xgb = None
    if HAS_XGB:
        xgb = XGBClassifier(n_estimators=n_estimators, max_depth=6, learning_rate=0.1, random_state=seed)
        xgb.fit(X_train, y_train)
        xgb_cv_acc, xgb_cv_f1 = _cv_scores(X_h, texts, y, _xgb_factory(n_estimators, seed), seed, use_tfidf, max_features)
        rows.append({"model": "XGBoost", "cv_acc": xgb_cv_acc, "cv_f1": xgb_cv_f1, **_test_metrics(xgb, X_test, y_test)})

    df = pd.DataFrame(rows)
    best = str(df.sort_values("cv_f1", ascending=False).iloc[0]["model"])

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "rf": rf,
        "xgb": xgb,
        "best_model": best,
        "feature_names": feature_names,
        "vectorizer": vectorizer,
        "use_tfidf": use_tfidf,
        "metrics": df.to_dict("records"),
    }
    joblib.dump(artifact, config.MODEL_PATH)

    print("\n===== 模型对比 =====")
    cols = ["model", "cv_acc", "cv_f1", "accuracy", "precision", "recall", "f1", "auc"]
    print(df[cols].round(4).to_string(index=False))
    print(f"\n最优模型（按 5 折 CV F1）：{best}")
    print(f"模型已保存到 {config.MODEL_PATH}\n")

    if plots:
        from visualize import generate_plots

        paths = generate_plots(rf, xgb, X_test, y_test, df, config.PLOTS_DIR)
        print("评估图已保存：")
        for p in paths:
            print(f"  - {p}")
        print()

    imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("===== 随机森林 Top 15 特征重要性 =====")
    print(imp.head(15).to_string())

    return artifact


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="训练并对比钓鱼邮件检测模型")
    parser.add_argument("--data", type=str, default="synthetic", help="synthetic | real | kaggle | raw | CSV 路径")
    parser.add_argument("--n-normal", type=int, default=250)
    parser.add_argument("--n-phishing", type=int, default=250)
    parser.add_argument("--no-dedup", action="store_true", help="真实数据不去重（指标会虚高）")
    parser.add_argument("--sample", type=int, default=None, help="kaggle/raw 数据抽样条数（默认全量）")
    parser.add_argument("--no-tfidf", action="store_true", help="禁用 TF-IDF，只用 30 个手写特征")
    parser.add_argument("--max-features", type=int, default=5000, help="TF-IDF 词表上限")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--no-plots", action="store_true", help="不生成评估图")
    args = parser.parse_args()

    records = load_records(
        args.data, n_normal=args.n_normal, n_phishing=args.n_phishing,
        seed=args.seed, dedup=not args.no_dedup, sample=args.sample,
    )
    run(
        records, seed=args.seed, test_size=args.test_size,
        n_estimators=args.n_estimators, use_tfidf=not args.no_tfidf,
        max_features=args.max_features, plots=not args.no_plots,
    )
