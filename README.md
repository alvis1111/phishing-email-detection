# 基于机器学习的钓鱼邮件检测系统

用 Python + 机器学习做二分类，识别钓鱼邮件。从**发件人、正文、URL** 三类维度提取特征，
训练 **随机森林** 与 **XGBoost** 两个模型并对比，最终输出**风险评分**与可解释的触发信号。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 一键跑通（合成数据：生成 -> 训练对比 -> 示例预测）
python main.py demo
```

分步执行：

```bash
python main.py gen-data              # 生成合成演示数据 data/emails.csv
python main.py train                 # 用合成数据训练，保存到 models/model_artifact.joblib
python main.py predict --body "..."  # 预测单封邮件
```

## 接入真实数据

```bash
python main.py download-data         # 下载内置公开数据集（Zenodo，2000 封）
python main.py train --data real     # 用它训练（默认去重 + 5 折交叉验证）
python main.py train --data kaggle   # 用 Kaggle phishing_email 数据集（约 8.2 万封，自动下载）
python main.py train --data kaggle --sample 20000   # 抽样 2 万条快速验证
python main.py train --data raw      # 用带原始正文/URL 的 Kaggle 18k 数据集
python main.py train --data my.csv   # 或直接指定你自己的 CSV
```

也可以直接跑 `python data_loader.py` 查看内置数据集情况。

### 内置数据集与去重

内置的 Zenodo「Phishing Validation Emails」数据集有 2000 封邮件（正常 / 钓鱼各 1000），
但**其中只有 100 个唯一文本**（其余 1900 行是模板复制）。不先去重的话，同一个模板会
同时进入训练集和测试集，交叉验证指标会虚高到 1.0。

因此默认**按正文去重**（2000 -> 100 条，钓鱼 23 / 正常 77），并用 **5 折交叉验证**给出诚实的泛化指标：

| 模型 | 5 折 CV 准确率 | 5 折 CV F1 |
|------|---------------|-----------|
| 随机森林 | 0.91 | 0.75 |
| XGBoost | 0.93 | 0.78 |

> 结论：这份公开数据集样本太少、模板化严重，训练出的模型对任意手写邮件泛化有限。
> 它适合跑通流程与验证方法论，不适合作为生产级数据。要更好的效果，用下面的 Kaggle 数据集。

### 接入 Kaggle phishing_email 数据集（约 8.2 万封）

直接执行即可自动下载并训练（公开数据，无需 Kaggle token）：

```bash
python main.py train --data kaggle                # 全量 8.2 万封
python main.py train --data kaggle --sample 20000 # 抽样 2 万条快速验证
```

来源：[Kaggle - Phishing Email Dataset（naserabdullahalam）](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)，
合并了 Enron / Ling / CEAS / Nazario / Nigerian Fraud / SpamAssassin 六个来源，共 **82,486 封**
（钓鱼+垃圾 42,891 / 正常 39,595），列 `text_combined` + `label`。

**实测结果（5 折交叉验证）：**

| 特征 | 随机森林 F1 | XGBoost F1 |
|------|------------|------------|
| 仅 30 个手写特征（抽样 2 万） | 0.729 | 0.754 |
| 手写特征 + TF-IDF（抽样 1 万） | 0.970 | 0.967 |

叠加 TF-IDF 后 F1 从约 0.75 提到约 0.95，且 `body_len` 的特征重要性从 0.57 降到 0.02——
长度偏差被化解，模型真正靠语义词（`enron` / `thanks` / `money` / `mailing` …）区分。

**两点需要知道：**

1. **该数据的 `text_combined` 是预处理过的文本**（已小写、去标点、无 URL、无标准 From/Subject 头），
   所以发件人 `snd_` 与 URL `url_` 特征基本用不上，TF-IDF 主要从正文词袋取信号。
   要启用发件人/URL 特征，需用带原始头部的邮件数据（如 `--data synthetic` 的合成数据）。

2. **长度偏差**：正常邮件平均 203 词、钓鱼/垃圾 121 词，仅靠词数逻辑回归就有 F1 0.64。
   不加 TF-IDF 时约 0.75 的 F1 里有相当一部分来自「短 = 钓鱼」；叠加 TF-IDF 后长度不再是主导特征。

### 带原始正文/URL 的数据（`--data raw`）

`--data raw` 用 [Kaggle - subhajournal/phishingemails](https://www.kaggle.com/datasets/subhajournal/phishingemails)
（`Phishing_Email.csv`，18,650 封，Safe 11,322 / Phishing 7,328），其 `Email Text` 相比上一份
保留了更多原始信息：**约 35% 的邮件含真实 URL**、约 40% 含邮箱地址、少量含 `From:` 头。

`parse_raw_email` 会解析头部，并在无 `From:` 头时把正文里第一个邮箱地址作为发件人兜底，
让 `snd_`、`url_` 特征在这份数据上也能启用。

**实测（抽样 1.2 万条，5 折 CV）：RF F1 0.948 / XGBoost 0.947。** 其中 `body_exclamation_count`
（感叹号数）进入 Top 特征——垃圾邮件爱用「!!!」，手写特征在这份数据上确有贡献。

**如实说明（免费公开数据集的普遍局限）：**

- 这份数据的「Phishing」多是**通用垃圾邮件**（成人/保险/尼日利亚诈骗），URL 是普通
  `.net`/`.com` 域名，并非品牌仿冒；所以 `url_has_lookalike`、`url_has_suspicious_tld`
  几乎为 0，`url_` 特征点燃 ~20% 但区分度有限。
- 真正带完整头部、含品牌仿冒 URL 的邮件数据集几乎都需付费或需自行抓取；免费数据集
  普遍已把头部剥掉。要完整演示发件人/URL 特征，`--data synthetic` 的合成数据是最可控的选择。

### 自定义 CSV 格式

`data_loader.py` 会自动识别「文本列 + 标签列」。最直接的格式是两列：

```
Email Text,Email Type
"Dear customer, verify your account now...",Phishing Email
"Hi Alex, your order has shipped...",Safe Email
```

也可以带 `sender` / `subject` / `body` 三列（会自动取文本列作正文，且用 `parse_raw_email`
把含 From:/Subject: 头的原始文本拆成发件人/主题/正文）。
标签列含 `phishing` / `spam` / `fraud` / `scam` 记为 1（恶意），其余记为 0。

## 项目结构

```
phishing-email-detection/
├── config.py            # 路径与常量（含 UTF-8 控制台修复）
├── data_synthesis.py    # 合成演示数据生成器（正常 / 钓鱼）
├── data_loader.py       # 真实数据下载 + CSV 自动解析
├── features.py          # 特征提取：发件人 + 正文 + URL + TF-IDF
├── train.py             # 训练 RF + XGBoost，CV + 测试集对比，保存模型
├── visualize.py         # 评估可视化：ROC / 混淆矩阵 / CV 对比
├── explain.py           # SHAP 特征归因：解释单封邮件的判定依据
├── predict.py           # 单封邮件预测，输出风险评分与触发信号
├── main.py              # 统一 CLI 入口
├── data/                # 数据（emails.csv、phishing_email.csv 等）
├── models/              # 训练产物（model_artifact.joblib）
└── plots/               # 评估图（roc_curve / confusion_matrix / cv_comparison）
```

## 特征设计

### 手写特征（30 个）

| 类别 | 前缀 | 示例特征 |
|------|------|----------|
| 发件人 | `snd_` | 是否免费邮箱、域名长度/数字/连字符、仿冒品牌域名、显示名与邮箱不符 |
| 正文 | `body_` | 长度、词数、大写占比、感叹号数、紧迫/威胁/中奖/凭证/时间压力关键词、泛化称呼、Click here |
| URL | `url_` | 链接数量、IP 地址、@ 符号、http 明文、域名点数/长度、仿冒域名、可疑 TLD、诱导路径 |

### TF-IDF 文本特征（默认开启，`tfidf_` 前缀）

对「主题 + 正文」做 TF-IDF 词袋（1-2 gram，词表默认 5000），与手写特征拼接。
可调参数：`--max-features`（词表大小）、`--no-tfidf`（关闭，只用 30 个手写特征）。

> 内置 Zenodo 数据只有正文文本，无发件人/URL 字段；Kaggle 数据的 `text_combined` 也已被
> 预处理（无 URL、无头部）。因此这两类数据训练时 `snd_`、`url_` 特征恒为 0，模型主要依赖
> `body_` 与 TF-IDF 特征。合成数据（`--data synthetic`）三类特征齐全，适合演示完整能力。
>
> 合成数据覆盖 10 种钓鱼场景（账户停用、假发票、包裹派送、密码过期、异常登录、中奖、
> 退税、订阅续费、勒索、彩票）与 7 种正常场景，并内置多种发件人伪造（仿冒域名、免费邮箱、
> 显示名伪装）与 URL 欺骗手法（IP、@、子域名、短链、可疑 TLD），可充分演示发件人/URL 特征。

## 风险评分

模型输出钓鱼概率 `p`，映射为 0-100 风险分：

- `p >= 0.7` -> 高风险（钓鱼）
- `0.3 <= p < 0.7` -> 中风险（需人工复核）
- `p < 0.3` -> 低风险（正常）

同时列出命中的可疑信号（如「URL 使用 IP 地址」「发件人域名仿冒知名品牌」），便于人工判断。

## 评估可视化

训练时自动在 `plots/` 下生成三张评估图（`--no-plots` 可关闭）：

- **`roc_curve.png`** — RF 与 XGBoost 的 ROC 曲线 + AUC
- **`confusion_matrix.png`** — 两模型的混淆矩阵（正常 / 钓鱼）
- **`cv_comparison.png`** — 交叉验证 F1 vs 测试集 F1 / 准确率 / AUC 的对比

交叉验证（5 折、每折重拟合 TF-IDF 词表）给出的是无数据泄漏的泛化估计；
测试集指标与 CV 的差距越小，说明模型越稳定。

## SHAP 特征归因

预测时自动用 SHAP（`TreeExplainer`）解释判定依据——列出把预测**推向钓鱼**与**推向正常**的
Top 特征及其贡献值，并保存发散条形图 `plots/shap_explanation.png`：

```bash
python main.py predict --body "..."          # 预测后自动输出 SHAP 归因
python main.py predict --body "..." --shap-top 15   # 展示 Top 15
python main.py predict --body "..." --no-shap       # 关闭（shap 未安装时自动跳过）
```

> 需安装 `shap`（`pip install shap`）；未安装时自动降级跳过，不影响预测。

## 常见问题

- **未安装 xgboost**：`train.py` 会自动降级为只训练随机森林并给出提示；安装后重新运行即可。
- **中文乱码**：已在 `config.py` 中强制 UTF-8 输出；旧 cmd/PowerShell 若仍乱码，先执行 `chcp 65001`。
- **扩展特征**：在 `features.py` 的三个 `extract_*_features` 中新增键即可，`FEATURE_NAMES` 自动更新；
  如需展示说明，在 `SIGNALS` 列表补充对应条目。
