"""混合真实数据 + 合成仿冒数据训练，补足品牌仿冒 / 可疑 TLD 等真钓鱼信号。

原因：免费公开真实数据集里的「钓鱼」多为通用垃圾邮件（普通 .net/.com 域名），
几乎没有品牌仿冒样本；而 AI 钓鱼恰恰靠仿冒域名 + 诱导链接。把内置的合成
仿冒场景混进来，让模型学会给 snd_/url_ 仿冒信号加权重。

用法：
    python train_mixed.py

合成数量可用下方 SYN_NORMAL / SYN_PHISH 调整（默认各 2000，约占总量 18%）。
"""
from __future__ import annotations

import config
from data_loader import load_records
from data_synthesis import generate_dataset
from train import run

REAL_SOURCE = "data/phishing_email.csv"  # 18,650 封真实邮件
SYN_NORMAL = 2000
SYN_PHISH = 2000


def main() -> None:
    real = load_records(REAL_SOURCE)
    synth = generate_dataset(SYN_NORMAL, SYN_PHISH, seed=config.SEED)
    records = real + synth
    n_phish = sum(r["label"] for r in records)
    print(f"真实 {len(real)} 封 + 合成 {len(synth)} 封 = 共 {len(records)} 封（钓鱼 {n_phish} / 正常 {len(records) - n_phish}）")
    run(records)


if __name__ == "__main__":
    main()
