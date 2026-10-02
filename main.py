"""统一入口。

用法：
    python main.py gen-data                 # 生成合成演示数据
    python main.py train                    # 用合成数据训练
    python main.py train --data real        # 用内置真实数据训练
    python main.py train --data my.csv      # 用自定义 CSV 训练
    python main.py download-data            # 下载内置真实数据集
    python main.py predict --body "..."     # 预测单封邮件
    python main.py demo                     # 一键跑通：生成 + 训练 + 示例预测
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="基于机器学习的钓鱼邮件检测系统")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("gen-data", help="生成合成演示数据")
    p_gen.add_argument("--n-normal", type=int, default=250)
    p_gen.add_argument("--n-phishing", type=int, default=250)
    p_gen.add_argument("--seed", type=int, default=42)

    p_train = sub.add_parser("train", help="训练并对比模型")
    p_train.add_argument("--data", type=str, default="synthetic", help="synthetic | real | kaggle | raw | CSV 路径")
    p_train.add_argument("--n-normal", type=int, default=250)
    p_train.add_argument("--n-phishing", type=int, default=250)
    p_train.add_argument("--no-dedup", action="store_true", help="真实数据不去重")
    p_train.add_argument("--sample", type=int, default=None, help="kaggle 数据抽样条数（默认全量）")
    p_train.add_argument("--no-tfidf", action="store_true", help="禁用 TF-IDF，只用 30 个手写特征")
    p_train.add_argument("--max-features", type=int, default=5000, help="TF-IDF 词表上限")
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--test-size", type=float, default=0.25)
    p_train.add_argument("--n-estimators", type=int, default=200)
    p_train.add_argument("--no-plots", action="store_true", help="不生成评估图")

    p_pred = sub.add_parser("predict", help="预测单封邮件")
    p_pred.add_argument("--sender", type=str, default="")
    p_pred.add_argument("--subject", type=str, default="")
    p_pred.add_argument("--body", type=str, default=None)
    p_pred.add_argument("--file", type=str, default=None, help="JSON 文件路径，含 sender/subject/body 字段")
    p_pred.add_argument("--shap-top", type=int, default=10, help="SHAP 展示的 Top 特征数")
    p_pred.add_argument("--no-shap", action="store_true", help="跳过 SHAP 特征归因")

    sub.add_parser("download-data", help="下载内置真实数据集（Zenodo）")
    sub.add_parser("demo", help="一键跑通：生成数据 + 训练 + 示例预测")

    args = parser.parse_args()

    if args.cmd == "gen-data":
        import config
        from data_synthesis import generate_dataset, save_dataset

        records = generate_dataset(args.n_normal, args.n_phishing, args.seed)
        save_dataset(records, config.DATA_PATH)

    elif args.cmd == "download-data":
        from data_loader import download_real_dataset

        download_real_dataset()

    elif args.cmd == "train":
        from data_loader import load_records
        from train import run

        records = load_records(args.data, n_normal=args.n_normal, n_phishing=args.n_phishing, seed=args.seed, dedup=not args.no_dedup, sample=args.sample)
        run(records, seed=args.seed, test_size=args.test_size, n_estimators=args.n_estimators, use_tfidf=not args.no_tfidf, max_features=args.max_features, plots=not args.no_plots)

    elif args.cmd == "predict":
        from predict import load_artifact, predict_email

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
            from explain import explain_email

            explain_email(artifact, email, top_k=args.shap_top)

    elif args.cmd == "demo":
        from data_loader import load_records
        from data_synthesis import generate_dataset
        from train import run
        from predict import predict_email

        records = load_records("synthetic", n_normal=250, n_phishing=250, seed=42)
        artifact = run(records, seed=42)

        # 从生成器中各取一封真实的钓鱼 / 正常邮件作为示例（保证在训练分布内）
        samples = generate_dataset(10, 10, seed=123)
        phish = next(r for r in samples if r["label"] == 1)
        legit = next(r for r in samples if r["label"] == 0)

        print("\n===== 示例预测 =====")
        for email in (phish, legit):
            predict_email(artifact, email)
            print()


if __name__ == "__main__":
    main()
