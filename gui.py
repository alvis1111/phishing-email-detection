"""钓鱼邮件检测器 — 图形界面（免命令行）。

打包成 exe 后双击运行：选择邮件文件或文件夹，点「开始检测」即可，结果实时显示并保存 CSV。

开发环境运行：python gui.py
打包后自检：  PhishingDetector.exe --selftest
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import joblib
import pandas as pd

MODEL_REL = Path("models") / "model_artifact.joblib"


def resource_path(rel: Path | str = "") -> Path:
    """兼容 PyInstaller 打包：冻结后资源在 sys._MEIPASS，开发时在脚本目录。"""
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS)
    else:
        base = Path(__file__).resolve().parent
    return base / rel if rel else base


def load_model() -> dict:
    path = resource_path(MODEL_REL)
    if not path.exists():
        raise FileNotFoundError(f"未找到模型文件：{path}")
    return joblib.load(path)


def collect(target: Path) -> list[tuple[str, dict]]:
    """把文件/文件夹统一成 (标签, email) 列表；文件夹递归扫描 .eml/.txt/.mbox。"""
    from batch_predict import EMAIL_EXT, iter_mbox, parse_email_file

    if target.is_file():
        if target.suffix.lower() == ".mbox":
            return list(iter_mbox(target))
        return [(target.name, parse_email_file(target))]

    items: list[tuple[str, dict]] = []
    for f in sorted(target.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in EMAIL_EXT:
            continue
        if f.suffix.lower() == ".mbox":
            items.extend(iter_mbox(f))
        else:
            items.append((str(f.relative_to(target)), parse_email_file(f)))
    return items


def detect(target: Path, artifact: dict) -> list[dict]:
    """批量判定，返回结果行列表（按风险分降序）。"""
    from features import build_full_features, explain_signals, extract_features
    from predict import apply_risk_rules, risk_level

    items = collect(target)
    if not items:
        return []

    emails = [e for _, e in items]
    vectorizer = artifact.get("vectorizer")
    if vectorizer is not None:
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
        prob, _ = apply_risk_rules(float(p), feats)
        signals = explain_signals(feats)
        rows.append(
            {
                "file": label,
                "sender": email.get("sender", ""),
                "subject": email.get("subject", ""),
                "prob": round(prob, 4),
                "score": round(prob * 100),
                "risk": risk_level(prob),
                "signals": "；".join(signals) if signals else "",
            }
        )
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


def _selftest() -> int:
    """造一封仿冒钓鱼邮件，跑通「模型加载 + 解析 + 判定」全链路（供打包后无界面验证）。"""
    import tempfile

    from batch_predict import parse_email_file

    sample = (
        "From: \"Microsoft Account Team\" <security@microsoft-security-alert.com>\r\n"
        "To: victim@example.com\r\n"
        "Subject: Your account has been suspended - action required\r\n"
        "Content-Type: text/plain; charset=\"utf-8\"\r\n"
        "\r\n"
        "Dear Customer,\r\n\r\n"
        "Your Microsoft account has been suspended due to unusual sign-in.\r\n"
        "Verify immediately: http://192.168.1.100/login\r\n"
    )
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "sample.eml"
        p.write_bytes(sample.encode("utf-8"))
        artifact = load_model()
        rows = detect(p, artifact)
        if not rows:
            print("SELFTEST FAIL: no rows")
            return 1
        r = rows[0]
        print(f"SELFTEST OK: score={r['score']} risk={r['risk']} file={r['file']}")
        return 0


# ---------------- GUI ----------------

def _build_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("钓鱼邮件检测器")
    root.geometry("960x600")
    root.minsize(760, 480)

    state = {"artifact": None, "target": None, "rows": []}

    # 顶部控制栏
    top = ttk.Frame(root, padding=10)
    top.pack(fill="x")

    ttk.Label(top, text="选择要检测的邮件文件或文件夹，然后点「开始检测」：").pack(anchor="w")

    ctrl = ttk.Frame(top)
    ctrl.pack(fill="x", pady=8)
    ttk.Button(ctrl, text="选择文件…", command=lambda: _pick_file(root, state)).pack(side="left")
    ttk.Button(ctrl, text="选择文件夹…", command=lambda: _pick_dir(root, state)).pack(side="left", padx=(6, 0))
    ttk.Button(ctrl, text="开始检测", command=lambda: _start(root, state)).pack(side="left", padx=(16, 0))
    ttk.Button(ctrl, text="导出 CSV…", command=lambda: _export(state)).pack(side="left", padx=(6, 0))

    path_lbl = ttk.Label(ctrl, text="（未选择）", foreground="#666")
    path_lbl.pack(side="left", padx=(12, 0))

    # 结果表
    table_frame = ttk.Frame(root, padding=(10, 0))
    table_frame.pack(fill="both", expand=True)

    cols = ("file", "score", "risk", "sender", "subject")
    heads = {"file": "文件", "score": "风险分", "risk": "风险等级", "sender": "发件人", "subject": "主题"}
    tree = ttk.Treeview(table_frame, columns=cols, show="headings", selectmode="browse")
    for c in cols:
        tree.heading(c, text=heads[c])
    tree.column("file", width=200, anchor="w")
    tree.column("score", width=70, anchor="center")
    tree.column("risk", width=130, anchor="w")
    tree.column("sender", width=220, anchor="w")
    tree.column("subject", width=280, anchor="w")

    tree.tag_configure("high", foreground="#c0392b")
    tree.tag_configure("mid", foreground="#e67e22")
    tree.tag_configure("low", foreground="#1e8449")

    vsb = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
    hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
    tree.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    hsb.grid(row=1, column=0, sticky="ew")
    table_frame.rowconfigure(0, weight=1)
    table_frame.columnconfigure(0, weight=1)

    # 选中行的触发信号
    detail = ttk.LabelFrame(root, text="触发信号", padding=8)
    detail.pack(fill="x", padx=10, pady=(0, 6))
    sig_text = tk.Text(detail, height=3, wrap="word", state="disabled")
    sig_text.pack(fill="x")

    def on_select(_evt=None):
        sel = tree.selection()
        if not sel:
            return
        idx = tree.index(sel[0])
        if idx < len(state["rows"]):
            r = state["rows"][idx]
            sig_text.configure(state="normal")
            sig_text.delete("1.0", "end")
            sig_text.insert("1.0", r["signals"] or "（无明显可疑信号）")
            sig_text.configure(state="disabled")

    tree.bind("<<TreeviewSelect>>", on_select)

    status = ttk.Label(root, text="就绪", foreground="#666", padding=(10, 4))
    status.pack(fill="x", side="bottom")

    def _pick_file(root, state):
        f = filedialog.askopenfilename(
            title="选择邮件文件",
            filetypes=[("邮件文件", "*.eml *.txt *.mbox"), ("所有文件", "*.*")],
        )
        if f:
            state["target"] = Path(f)
            state["rows"] = []
            path_lbl.configure(text=str(state["target"]))
            _clear_tree()

    def _pick_dir(root, state):
        d = filedialog.askdirectory(title="选择邮件文件夹")
        if d:
            state["target"] = Path(d)
            state["rows"] = []
            path_lbl.configure(text=str(state["target"]))
            _clear_tree()

    def _clear_tree():
        for i in tree.get_children():
            tree.delete(i)
        sig_text.configure(state="normal")
        sig_text.delete("1.0", "end")
        sig_text.configure(state="disabled")

    def _fill_tree():
        _clear_tree()
        for i, r in enumerate(state["rows"]):
            tag = "high" if r["score"] >= 70 else ("mid" if r["score"] >= 30 else "low")
            tree.insert("", "end", iid=str(i), values=(r["file"], r["score"], r["risk"], r["sender"], r["subject"]), tags=(tag,))

    def _start(root, state):
        if not state["target"]:
            messagebox.showwarning("提示", "请先选择邮件文件或文件夹。")
            return
        status.configure(text="正在加载模型并检测，请稍候…")
        for b in ctrl.winfo_children():
            if isinstance(b, ttk.Button):
                b.configure(state="disabled")

        def worker():
            try:
                if state["artifact"] is None:
                    state["artifact"] = load_model()
                rows = detect(state["target"], state["artifact"])
                err = None
            except Exception as e:  # noqa: BLE001
                rows, err = [], e
            root.after(0, lambda: _done(rows, err))

        threading.Thread(target=worker, daemon=True).start()

    def _done(rows, err):
        for b in ctrl.winfo_children():
            if isinstance(b, ttk.Button):
                b.configure(state="normal")
        if err is not None:
            status.configure(text="检测失败")
            messagebox.showerror("错误", f"检测失败：{err}")
            return
        state["rows"] = rows
        _fill_tree()
        n_high = sum(1 for r in rows if r["score"] >= 70)
        n_mid = sum(1 for r in rows if 30 <= r["score"] < 70)
        n_low = sum(1 for r in rows if r["score"] < 30)
        msg = f"共 {len(rows)} 封：高风险 {n_high} / 中风险 {n_mid} / 低风险 {n_low}"
        status.configure(text=msg)

        # 自动保存 CSV 到所选位置旁
        default_out = _default_out(state["target"])
        try:
            pd.DataFrame(rows)[["file", "sender", "subject", "score", "risk", "signals"]].to_csv(
                default_out, index=False, encoding="utf-8-sig"
            )
            status.configure(text=f"{msg}   ·  已保存 {default_out}")
        except Exception:  # noqa: BLE001
            pass

    def _default_out(target: Path) -> Path:
        base = target if target.is_dir() else target.parent
        return base / "检测结果.csv"

    def _export(state):
        if not state["rows"]:
            messagebox.showinfo("提示", "还没有结果，请先检测。")
            return
        f = filedialog.asksaveasfilename(
            title="导出 CSV",
            defaultextension=".csv",
            initialfile="检测结果.csv",
            filetypes=[("CSV 文件", "*.csv")],
        )
        if f:
            pd.DataFrame(state["rows"])[["file", "sender", "subject", "score", "risk", "signals"]].to_csv(
                f, index=False, encoding="utf-8-sig"
            )
            status.configure(text=f"已导出 {f}")

    root.mainloop()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    _build_gui()
