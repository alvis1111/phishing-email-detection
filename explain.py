"""SHAP 特征归因：解释单封邮件为什么被判为钓鱼 / 正常。

用 TreeExplainer 对树模型（随机森林 / XGBoost）算 SHAP 值，输出
「哪些特征把预测往钓鱼方向推、哪些往正常方向推」，并保存发散条形图。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from visualize import AXIS, BLUE, GRID, INK, RED, SURFACE

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# 手写特征 -> 可读中文名（SHAP 展示用）
FEATURE_LABELS = {
    "snd_is_free_email": "发件人用免费邮箱",
    "snd_domain_len": "发件人域名长度",
    "snd_domain_digits": "发件人域名含数字",
    "snd_domain_hyphens": "发件人域名含连字符",
    "snd_localpart_len": "发件人本地部分长度",
    "snd_localpart_digits": "发件人本地部分含数字",
    "snd_lookalike_brand": "发件人域名仿冒品牌",
    "snd_display_brand_mismatch": "发件人显示名与域名不符",
    "body_len": "正文长度",
    "body_word_count": "正文词数",
    "body_caps_ratio": "正文大写占比",
    "body_exclamation_count": "正文感叹号数",
    "body_urgency_count": "紧迫措辞次数",
    "body_generic_greeting": "泛化称呼",
    "body_has_click_here": "含 Click here",
    "body_has_verify": "要求验证",
    "body_has_credential": "索取凭证",
    "body_has_threat": "含威胁",
    "body_has_prize": "声称中奖",
    "body_has_time_pressure": "时间压力",
    "url_count": "URL 数量",
    "url_unique_domains": "不同域名数",
    "url_has_ip": "URL 用 IP 地址",
    "url_has_at": "URL 含 @ 符号",
    "url_has_http": "URL 用 http 明文",
    "url_max_dots": "域名最多点数",
    "url_max_domain_len": "域名最大长度",
    "url_has_lookalike": "URL 域名仿冒",
    "url_has_suspicious_tld": "URL 可疑 TLD",
    "url_has_suspicious_path": "URL 诱导路径",
}


def _label(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    if name.startswith("tfidf_"):
        return f"词「{name[len('tfidf_'):]}」"
    return name


def _build_sample(artifact: dict, email: dict):
    """构建单封邮件的稠密特征向量 + 特征名列表。"""
    from features import build_full_features, extract_features

    vectorizer = artifact.get("vectorizer")
    if vectorizer is not None:
        X, names, _ = build_full_features([email], vectorizer)
        return X.toarray(), names
    feats = extract_features(email)
    names = list(artifact.get("feature_names", [])) or list(feats.keys())
    X = pd.DataFrame([feats]).reindex(columns=names, fill_value=0).values
    return X, names


def _resolve_model(artifact: dict, model_name: str):
    if model_name == "XGBoost" and artifact.get("xgb") is not None:
        return artifact["xgb"]
    return artifact["rf"]


def explain_email(
    artifact: dict,
    email: dict,
    model_name: str | None = None,
    top_k: int = 10,
    plot_path: Path | str | None = None,
) -> pd.Series | None:
    """用 SHAP 解释单封邮件预测，打印 Top 特征并保存归因图。

    返回 SHAP 值 Series（index=特征名）；shap 未安装时返回 None。
    """
    if not HAS_SHAP:
        print("[提示] 未安装 shap，跳过特征归因。可执行: pip install shap")
        return None

    model_name = model_name or artifact.get("best_model", "RandomForest")
    model = _resolve_model(artifact, model_name)

    X, names = _build_sample(artifact, email)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    # 不同 shap 版本返回结构不一，统一取「样本 0 的类别 1（钓鱼）」SHAP 向量：
    #   - list            -> [class0, class1]，取 class1 的样本 0
    #   - (n, f, 2) 三维   -> 取样本 0、类别 1（shap 0.5x 的 ndarray）
    #   - (n, f) / (1, f)  -> 单输出（回归 / 二分类单向量），取样本 0
    if isinstance(shap_vals, list):
        src = np.asarray(shap_vals[1]) if len(shap_vals) > 1 else np.asarray(shap_vals[0])
        vals = src[0]
    else:
        arr = np.asarray(shap_vals)
        if arr.ndim == 3:
            cls = 1 if arr.shape[2] > 1 else 0
            vals = arr[0, :, cls]
        elif arr.ndim == 2:
            vals = arr[0]
        else:
            vals = arr

    s = pd.Series(vals, index=names)

    print(f"\n===== SHAP 特征归因（模型 {model_name}）=====")
    top = s.reindex(s.abs().sort_values(ascending=False).index[:top_k])
    for name, v in top.items():
        direction = "→ 钓鱼" if v >= 0 else "→ 正常"
        print(f"  {_label(name):<28s} {v:+.4f}  {direction}")

    if plot_path is None:
        plot_path = config.PLOTS_DIR / "shap_explanation.png"
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_bar(top, plot_path)
    print(f"归因图已保存到 {plot_path}")

    return s


def _plot_bar(top: pd.Series, out_path: Path) -> None:
    """发散条形图：正 SHAP（红，→钓鱼）在右，负（蓝，→正常）在左。"""
    top = top.sort_values()
    fig, ax = plt.subplots(figsize=(7.6, max(3.2, len(top) * 0.38)))
    colors = [RED if v >= 0 else BLUE for v in top.values]
    ax.barh(range(len(top)), top.values, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([_label(n) for n in top.index], fontsize=9)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_xlabel("SHAP 值（正 → 钓鱼，负 → 正常）")
    ax.set_title("SHAP 特征归因：单封邮件")
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(color=GRID, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
