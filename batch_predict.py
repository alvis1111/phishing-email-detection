"""批量预测：对邮件文件批量判定是否钓鱼。

支持三种输入：
    - 文件夹：扫描其中的 .eml / .txt / .mbox
    - 单个 .eml / .txt 文件
    - 单个 .mbox 文件（Gmail「Google 导出/Takeout」的格式，一个文件装全部邮件）

用法：
    python batch_predict.py <文件夹或文件路径> [--out 结果CSV路径] [--limit N]

行为：
    - .eml 用 Python 标准库 email 模块解析（自动解码 base64 / quoted-printable，
      提取 From / Subject 与纯文本正文；纯 HTML 邮件会剥掉标签并保留 href 里的 URL）。
    - .mbox 用 mailbox 模块逐封解析（同一套解码逻辑）。
    - .txt 按原始文本读取，交给 parse_raw_email 拆头部。
    - 复用 models/model_artifact.joblib 里的模型，一次向量化全部邮件后批量打分。
    - 按风险分从高到低打印汇总，并保存 CSV（utf-8-sig，Excel 可直接打开）。

输出 CSV 列：file / sender / subject / prob / score / risk / signals
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

import config
from features import extract_features, explain_signals, parse_raw_email
from predict import apply_risk_rules, load_artifact, risk_level

EMAIL_EXT = {".eml", ".txt", ".mbox"}

_TAG_RE = re.compile(r"<[^>]+>")
_HREF_RE = re.compile(r'href\s*=\s*["\']?(https?://[^\s"\'<>]+)', re.IGNORECASE)
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)


def _html_to_text(html: str) -> str:
    """把 HTML 正文粗转成纯文本，并保留 href 里的 URL（URL 特征依赖它）。"""
    html = _SCRIPT_STYLE_RE.sub(" ", html)
    urls = _HREF_RE.findall(html)
    text = _TAG_RE.sub(" ", html)
    text = re.sub(r"\s+", " ", text).strip()
    if urls:
        text = (text + "\n" + "\n".join(urls)).strip()
    return text


def _message_to_email(msg) -> dict:
    """从 policy.default 解析出的 email.Message 提取 sender/subject/body。"""
    sender = str(msg.get("From", "") or "")
    subject = str(msg.get("Subject", "") or "")

    plain_parts: list[str] = []
    html_parts: list[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain":
                try:
                    plain_parts.append(part.get_content())
                except Exception:
                    pass
            elif ctype == "text/html":
                try:
                    html_parts.append(part.get_content())
                except Exception:
                    pass
    else:
        try:
            ctype = msg.get_content_type()
            content = msg.get_content()
        except Exception:
            ctype, content = "", ""
        if ctype == "text/html":
            html_parts.append(content)
        else:
            plain_parts.append(content)

    if plain_parts:
        body = "\n".join(p.strip() for p in plain_parts if p.strip()).strip()
    elif html_parts:
        body = "\n".join(_html_to_text(h) for h in html_parts if h).strip()
    else:
        body = ""
    return {"sender": sender, "subject": subject, "body": body}


def parse_email_file(path: Path) -> dict:
    """把 .eml / .txt 读成统一 email dict（sender/subject/body）。"""
    if path.suffix.lower() == ".eml":
        from email import policy
        from email.parser import BytesParser

        msg = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
        return _message_to_email(msg)

    # .txt 及其它：按原始文本读入，交给 parse_raw_email 拆 From:/Subject: 头
    raw = path.read_text(encoding="utf-8", errors="replace")
    return parse_raw_email(raw)


def iter_mbox(path: Path):
    """逐封解析 .mbox 文件，产出 (label, email_dict)。label 用「文件名#序号」。"""
    import mailbox
    from email import policy
    from email.parser import BytesParser

    mbox = mailbox.mbox(str(path), create=False, factory=lambda f: BytesParser(policy=policy.default).parse(f))
    for i, msg in enumerate(mbox, start=1):
        try:
            yield f"{path.name}#{i:04d}", _message_to_email(msg)
        except Exception:
            # 单封解析失败不影响整体
            continue


def collect_emails(target: Path) -> list[tuple[str, dict]]:
    """把目标（文件夹 / 单个 .eml/.txt/.mbox 文件）转成 (label, email) 列表。"""
    if target.is_dir():
        files = sorted(f for f in target.iterdir() if f.suffix.lower() in EMAIL_EXT)
        items: list[tuple[str, dict]] = []
        for f in files:
            if f.suffix.lower() == ".mbox":
                items.extend(iter_mbox(f))
            else:
                items.append((f.name, parse_email_file(f)))
        return items

    if target.suffix.lower() == ".mbox":
        return list(iter_mbox(target))
    return [(target.name, parse_email_file(target))]


def main() -> None:
    parser = argparse.ArgumentParser(description="批量判定邮件是否钓鱼（支持 .eml/.txt/.mbox）")
    parser.add_argument("target", help="邮件文件夹路径，或单个 .eml/.txt/.mbox 文件")
    parser.add_argument("--out", default=str(config.DATA_DIR / "batch_predictions.csv"), help="结果 CSV 输出路径")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 封（用于先试跑）")
    args = parser.parse_args()

    target = Path(args.target)
    if not target.exists():
        sys.exit(f"[错误] 找不到路径: {target}")

    items = collect_emails(target)
    if not items:
        sys.exit(f"[错误] 没有可处理的邮件（支持 .eml / .txt / .mbox）: {target}")
    if args.limit:
        items = items[: args.limit]

    artifact = load_artifact()
    emails = [email for _, email in items]

    # 一次向量化全部邮件（与 predict.py / train.py 同一套特征提取，保证特征列一致）
    vectorizer = artifact.get("vectorizer")
    if vectorizer is not None:
        from features import build_full_features

        X, _, _ = build_full_features(emails, vectorizer)
    else:
        feature_names = artifact["feature_names"]
        X = pd.DataFrame([extract_features(e) for e in emails]).reindex(columns=feature_names, fill_value=0)

    rf_probs = artifact["rf"].predict_proba(X)[:, 1]
    xgb = artifact.get("xgb")
    xgb_probs = xgb.predict_proba(X)[:, 1] if xgb is not None else None
    best = artifact.get("best_model", "RandomForest")
    probs = xgb_probs if (best == "XGBoost" and xgb_probs is not None) else rf_probs

    rows = []
    for (label, email), p in zip(items, probs):
        feats = extract_features(email)
        prob, _escalated = apply_risk_rules(float(p), feats)
        signals = explain_signals(feats)
        rows.append(
            {
                "file": label,
                "sender": email.get("sender", ""),
                "subject": email.get("subject", ""),
                "prob": prob,
                "score": round(prob * 100),
                "risk": risk_level(prob),
                "signals": "；".join(signals) if signals else "",
            }
        )

    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"共判定 {len(df)} 封邮件，结果已保存到 {out_path}\n")
    for _, r in df.iterrows():
        print(f"[{r['score']:>3}分] {r['risk']:<12s} {r['file']}")
        if r["subject"]:
            print(f"        主题: {r['subject']}")
        if r["sender"]:
            print(f"        发件人: {r['sender']}")
        if r["signals"]:
            print(f"        信号: {r['signals']}")
    print()
    n_high = int((df["prob"] >= 0.7).sum())
    n_mid = int(((df["prob"] >= 0.3) & (df["prob"] < 0.7)).sum())
    n_low = int((df["prob"] < 0.3).sum())
    print(f"汇总：高风险 {n_high} / 中风险 {n_mid} / 低风险 {n_low}")


if __name__ == "__main__":
    main()
