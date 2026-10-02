"""评估可视化：ROC 曲线、混淆矩阵、交叉验证对比。

颜色遵循 dataviz 规范：分类色用固定顺序（蓝 #2a78d6 / 橙 #eb6834），
混淆矩阵用蓝色顺序色阶（浅->深表示数量）。输出到 plots/ 目录。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无界面后端，直接写文件

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# 规范调色板（浅色表面）
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"     # 分类槽 1
ORANGE = "#eb6834"   # 分类槽 2
RED = "#e34948"      # 分类槽 8 / 发散色（SHAP 正向）
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281"]

plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "text.color": INK,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "xtick.color": SECONDARY,
        "ytick.color": SECONDARY,
    }
)


def _style_axes(ax, grid: bool = True) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(color=GRID, linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)


def plot_roc(curves: list[tuple[str, np.ndarray, np.ndarray, float]], out_path: Path) -> None:
    """curves: [(模型名, fpr, tpr, auc), ...]"""
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = [BLUE, ORANGE]
    for (name, fpr, tpr, auc), color in zip(curves, colors):
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name}（AUC = {auc:.3f}）")
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="随机基线")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("假阳性率 FPR")
    ax.set_ylabel("真阳性率 TPR")
    ax.set_title("ROC 曲线")
    ax.legend(frameon=False, loc="lower right")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(cms: list[tuple[str, np.ndarray]], out_path: Path, labels=("正常", "钓鱼")) -> None:
    """cms: [(模型名, 2x2 混淆矩阵), ...]"""
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.2), squeeze=False)
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    for ax, (name, cm) in zip(axes[0], cms):
        vmax = max(cm.max(), 1)
        ax.imshow(cm, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("预测类别")
        ax.set_ylabel("实际类别")
        ax.set_title(name)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                v = cm[i, j]
                color = "white" if v > vmax * 0.5 else INK
                ax.text(j, i, str(int(v)), ha="center", va="center", fontsize=13, color=color)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cv_comparison(metrics_df, out_path: Path) -> None:
    """metrics_df 需含列：model, cv_f1, f1, accuracy, auc。"""
    specs = [("cv_f1", "交叉验证 F1"), ("f1", "测试集 F1"), ("accuracy", "测试集准确率"), ("auc", "AUC")]
    models = metrics_df["model"].tolist()
    colors = {"RandomForest": BLUE, "XGBoost": ORANGE}
    x = np.arange(len(specs))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for i, m in enumerate(models):
        row = metrics_df[metrics_df["model"] == m].iloc[0]
        vals = [row[c] for c, _ in specs]
        off = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(x + off, vals, width, label=m, color=colors.get(m, BLUE))
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=INK,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in specs])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("得分")
    ax.set_title("模型对比：交叉验证 vs 测试集")
    ax.legend(frameon=False)
    _style_axes(ax, grid=False)
    ax.yaxis.grid(color=GRID, linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_plots(rf, xgb, X_test, y_test, metrics_df, out_dir: Path) -> list[Path]:
    """生成全部评估图，返回保存的文件路径列表。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [("RandomForest", rf)] + ([("XGBoost", xgb)] if xgb is not None else [])
    y_test = np.asarray(y_test)

    curves, cms = [], []
    for name, m in models:
        proba = m.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        curves.append((name, fpr, tpr, roc_auc_score(y_test, proba)))
        pred = (proba >= 0.5).astype(int)
        cms.append((name, confusion_matrix(y_test, pred)))

    paths = []
    p = out_dir / "roc_curve.png"
    plot_roc(curves, p)
    paths.append(p)
    p = out_dir / "confusion_matrix.png"
    plot_confusion_matrices(cms, p)
    paths.append(p)
    p = out_dir / "cv_comparison.png"
    plot_cv_comparison(metrics_df, p)
    paths.append(p)
    return paths
