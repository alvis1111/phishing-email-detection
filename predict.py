"""单封邮件预测：输出是否为钓鱼 + 风险评分 + 触发信号。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

import config
from features import build_full_features, extract_features, explain_signals


def risk_level(prob: float) -> str:
    if prob >= 0.7:
        return "高风险（钓鱼）"
    if prob >= 0.3:
        return "中风险（需人工复核）"
    return "低风险（正常）"


# 强钓鱼信号：命中任意一条即视为高风险（规则兜底）。
# 这些是「结构性、正常邮件几乎不会出现」的信号，尤其 AI 品牌仿冒钓鱼常用；
# 免费公开训练数据缺品牌仿冒样本，模型对这些信号权重不足，故打分后追加规则。
ESCALATE_SIGNALS = [
    "snd_lookalike_brand",      # 发件人域名仿冒品牌
    "url_has_lookalike",        # URL 域名仿冒品牌
    "url_has_suspicious_tld",   # URL 可疑顶级域名（.xyz/.top/.online…）
    "url_has_ip",               # URL 使用裸 IP 地址
    "url_has_at",               # URL 含 @（http://官方域@仿冒域）
]


def apply_risk_rules(prob: float, feats: dict) -> tuple[float, bool]:
    """返回 (最终概率, 是否被强信号规则升级)。"""
    if any(feats.get(k) for k in ESCALATE_SIGNALS):
        return max(prob, 0.85), True
    return prob, False


def load_artifact(path: Path | str = config.MODEL_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        sys.exit(f"[错误] 未找到模型 {path}，请先运行: python train.py")
    return joblib.load(path)


def predict_email(artifact: dict, email: dict, verbose: bool = True) -> tuple[float, int]:
    """返回 (钓鱼概率, 风险评分 0-100)。"""
    feats = extract_features(email)
    vectorizer = artifact.get("vectorizer")
    if vectorizer is not None:
        X, _, _ = build_full_features([email], vectorizer)
    else:
        X = pd.DataFrame([feats]).reindex(columns=artifact["feature_names"], fill_value=0)

    rf = artifact["rf"]
    prob_rf = float(rf.predict_proba(X)[0, 1])

    prob_xgb = None
    if artifact.get("xgb") is not None:
        prob_xgb = float(artifact["xgb"].predict_proba(X)[0, 1])

    best = artifact.get("best_model", "RandomForest")
    if best == "XGBoost" and prob_xgb is not None:
        prob = prob_xgb
    else:
        prob = prob_rf
    prob, escalated = apply_risk_rules(prob, feats)
    score = round(prob * 100)

    if verbose:
        print("=" * 46)
        print("邮件主题:", email.get("subject", ""))
        print("发件人:  ", email.get("sender", ""))
        print("-" * 46)
        print(f"判定结果: {risk_level(prob)}")
        print(f"风险评分: {score} / 100")
        print(f"钓鱼概率: {prob:.3f}")
        if prob_xgb is not None:
            print(f"  RandomForest 概率: {prob_rf:.3f}")
            print(f"  XGBoost       概率: {prob_xgb:.3f}")
        if escalated:
            print("  [规则升级] 命中发件人/URL 仿冒或可疑链接强信号，风险上调为高风险")
        signals = explain_signals(feats)
        if signals:
            print("触发信号:")
            for s in signals:
                print(f"  - {s}")
        else:
            print("触发信号: 无明显可疑信号")
        print("=" * 46)

    return prob, score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预测单封邮件是否为钓鱼")
    parser.add_argument("--sender", type=str, default="")
    parser.add_argument("--subject", type=str, default="")
    parser.add_argument("--body", type=str, default=None)
    parser.add_argument("--file", type=str, default=None, help="JSON 文件路径，含 sender/subject/body 字段")
    parser.add_argument("--shap-top", type=int, default=10, help="SHAP 展示的 Top 特征数")
    parser.add_argument("--no-shap", action="store_true", help="跳过 SHAP 特征归因")
    args = parser.parse_args()

    artifact = load_artifact()
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            email = json.load(f)
    else:
        if not args.body:
            sys.exit("[错误] 请用 --body 提供正文，或用 --file 指定 JSON 文件")
        email = {"sender": args.sender, "subject": args.subject, "body": args.body}

    predict_email(artifact, email)
    if not args.no_shap:
        # 用 importlib 动态加载，避免打包工具按字节码把 explain -> shap -> torch 一并卷入
        import importlib

        explain_email = importlib.import_module("explain").explain_email
        explain_email(artifact, email, top_k=args.shap_top)
