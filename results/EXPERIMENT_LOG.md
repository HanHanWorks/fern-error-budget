# 实验数据记录（Experiment registry）

本文件由 `code/build_experiment_registry.py` 从各次运行产出的 `config.json` / `test_metrics.json` / `full_metrics.json` / jsonl 原始结果自动汇总，共 **52** 条记录。机器可读版本：`results/registry.csv`。

所有数字均直接读取原始指标文件，未经二次计算，因此不会与原始记录不一致。

## 记录明细

| id | kind | backbone | mode | img | extra | train_n | test_n | top1 | top3 | macro | specimen | field |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bioclip2_ft | finetune | bioclip2 | finetune | 224 |  | 4185 | 902 | 0.8914 | 0.969 | 0.8905 | 0.9046 | 0.8859 |
| bioclip2_ft_336 | finetune | bioclip2 | finetune | 336 |  | 4185 | 902 | 0.9013 | 0.9723 | 0.8957 | 0.9122 | 0.8969 |
| bioclip2_ft_hier | finetune | bioclip2 | finetune | 224 | hier | 4185 | 902 | 0.8614 | 0.9612 | 0.8504 | 0.8779 | 0.8547 |
| bioclip2_ft_lr5e5 | finetune | bioclip2 | finetune | 224 |  | 4185 | 902 | 0.8459 | 0.9534 | 0.8424 | 0.8855 | 0.8297 |
| bioclip2_linear | finetune | bioclip2 | linear | 224 |  | 4185 | 902 | 0.898 | 0.9767 | 0.8952 | 0.8969 | 0.8984 |
| bioclip2_linear_336 | finetune | bioclip2 | linear | 336 |  | 4185 | 902 | 0.9235 | 0.9778 | 0.9227 | 0.9198 | 0.925 |
| bioclip2_linear_336_s20261010 | finetune | bioclip2 | linear | 336 |  | 4185 | 902 | 0.9235 | 0.9778 | 0.9227 | 0.9198 | 0.925 |
| bioclip2_linear_336_s20261011 | finetune | bioclip2 | linear | 336 |  | 4185 | 902 | 0.9124 | 0.9789 | 0.9084 | 0.9084 | 0.9141 |
| bioclip2_linear_336_s20261012 | finetune | bioclip2 | linear | 336 |  | 4185 | 902 | 0.9058 | 0.9778 | 0.8999 | 0.9084 | 0.9047 |
| bioclip2_linear_bal | finetune | bioclip2 | linear | 224 | class-balance | 4185 | 902 | 0.898 | 0.969 | 0.8956 | 0.8893 | 0.9016 |
| dinov2_ft | finetune | dinov2 | finetune | 224 |  | 4185 | 902 | 0.8592 | 0.9512 | 0.8516 | 0.8702 | 0.8547 |
| dinov2_linear | finetune | dinov2 | linear | 224 |  | 4185 | 902 | 0.7927 | 0.9357 | 0.776 | 0.8015 | 0.7891 |
| dinov2_linear_336 | finetune | dinov2 | linear | 336 |  | 4185 | 902 | 0.8437 | 0.9412 | 0.8346 | 0.8626 | 0.8359 |
| dinov2_linear_336_s20261010 | finetune | dinov2 | linear | 336 |  | 4185 | 902 | 0.8437 | 0.9412 | 0.8346 | 0.8626 | 0.8359 |
| dinov2_linear_336_s20261011 | finetune | dinov2 | linear | 336 |  | 4185 | 902 | 0.837 | 0.9435 | 0.8294 | 0.8626 | 0.8266 |
| dinov2_linear_336_s20261012 | finetune | dinov2 | linear | 336 |  | 4185 | 902 | 0.8492 | 0.9468 | 0.8413 | 0.874 | 0.8391 |
| geo_ex-CN | geo-transfer | bioclip2 | linear | 336 | holdout=CN (train_n_species=146) | 3293 | 171 | 0.6959 | 0.8713 | 0.7392 | 0.7374 | 0.6389 |
| geo_ex-CN_region | geo-transfer | bioclip2 | linear | 336 | holdout=CN,TW,HK,MO (train_n_species=130) | 1902 | 367 | 0.6158 | 0.8447 | 0.6624 | 0.6842 | 0.5919 |
| geo_ex-JP | geo-transfer | bioclip2 | linear | 336 | holdout=JP (train_n_species=152) | 3901 | 59 | 0.7966 | 0.9831 | 0.8062 | 0.8 | 0.7949 |
| geo_ex-RU | geo-transfer | bioclip2 | linear | 336 | holdout=RU (train_n_species=152) | 4011 | 54 | 0.9259 | 0.9815 | 0.9226 | 1.0 | 0.9231 |
| geo_ex-TH | geo-transfer | bioclip2 | linear | 336 | holdout=TH (train_n_species=152) | 4029 | 31 | 0.6452 | 0.8065 | 0.6406 | 0.64 | 0.6667 |
| geo_ex-TW | geo-transfer | bioclip2 | linear | 336 | holdout=TW (train_n_species=145) | 2860 | 244 | 0.5615 | 0.7951 | 0.5966 | 1.0 | 0.5523 |
| geo_ex-US | geo-transfer | bioclip2 | linear | 336 | holdout=US (train_n_species=152) | 3963 | 47 | 0.9149 | 0.9787 | 0.9392 |  | 0.9149 |
| geo_ex-VN | geo-transfer | bioclip2 | linear | 336 | holdout=VN (train_n_species=151) | 4088 | 28 | 0.8214 | 0.9286 | 0.8095 | 0.7619 | 1.0 |
| xfer_field2spec | transfer | bioclip2 | linear | 336 | image_domain=specimen->image_domain=field | 1166 | 156 | 0.6538 | 0.8013 | 0.726 |  | 0.6538 |
| xfer_gbif2inat | transfer | bioclip2 | linear | 336 | source=GBIF->source=iNaturalist | 1965 | 128 | 0.7891 | 0.9062 | 0.8071 |  | 0.7891 |
| xfer_inat2gbif | transfer | bioclip2 | linear | 336 | source=iNaturalist->source=GBIF | 2220 | 270 | 0.5 | 0.7111 | 0.5697 | 0.3395 | 0.7407 |
| xfer_spec2field | transfer | bioclip2 | linear | 336 | image_domain=field->image_domain=specimen | 3019 | 210 | 0.4238 | 0.6286 | 0.4285 | 0.4238 |  |
| zeroshot_bioclip2 | zero-shot | bioclip2 | zero-shot | 224 | no training | 0 | 902 | 0.6829 | 0.8381 | 0.6864 | 0.6069 | 0.7141 |
| bioclip2_ft_openset | open-set |  | max-softmax |  | AUROC=0.9701 TPR@FPR10=0.8758 n_neg=700 |  | 902 |  |  |  |  |  |
| bioclip2_ft_336_openset | open-set |  | max-softmax |  | AUROC=0.9666 TPR@FPR10=0.8858 n_neg=700 |  | 902 |  |  |  |  |  |
| bioclip2_ft_hier_openset | open-set |  | max-softmax |  | AUROC=0.9236 TPR@FPR10=0.7483 n_neg=700 |  | 902 |  |  |  |  |  |
| bioclip2_linear_openset | open-set |  | max-softmax |  | AUROC=0.9634 TPR@FPR10=0.9047 n_neg=700 |  | 902 |  |  |  |  |  |
| bioclip2_linear_336_openset | open-set |  | max-softmax |  | AUROC=0.9666 TPR@FPR10=0.9302 n_neg=700 |  | 902 |  |  |  |  |  |
| dinov2_ft_openset | open-set |  | max-softmax |  | AUROC=0.9425 TPR@FPR10=0.8204 n_neg=700 |  | 902 |  |  |  |  |  |
| dinov2_linear_336_openset | open-set |  | max-softmax |  | AUROC=0.9180 TPR@FPR10=0.7805 n_neg=700 |  | 902 |  |  |  |  |  |
| bioclip2_ft_336_full | derived-full |  | eval_full.py |  | family=0.9678 fam_macro=0.9617 ece=0.0954 lat=24.03ms rc@cov0.25=0.9911 |  | 902 | 0.9013 | 0.9723 |  |  |  |
| bioclip2_linear_336_full | derived-full |  | eval_full.py |  | family=0.9712 fam_macro=0.96 ece=0.1948 lat=22.55ms rc@cov0.25=1.0 |  | 902 | 0.9224 | 0.9778 |  |  |  |
| dinov2_linear_336_full | derived-full |  | eval_full.py |  | family=0.9324 fam_macro=0.9204 ece=0.1971 lat=20.97ms rc@cov0.25=0.9956 |  | 902 | 0.8448 | 0.9412 |  |  |  |
| plantnet_all_test | baseline-api | Pl@ntNet | zero-shot API |  | 902 ok / 8 rejected, 2 downscaled | 0 | 902 | 0.2428 |  |  |  |  |
| plantnet_k-china_test | baseline-api | Pl@ntNet | zero-shot API |  | 60 ok / 0 rejected | 0 | 60 | 0.35 |  |  |  |  |
| plantnet_k-world-flora_test | baseline-api | Pl@ntNet | zero-shot API |  | 60 ok / 0 rejected | 0 | 60 | 0.3333 |  |  |  |  |
| vlm_qwen3-vl-8b-instruct_family30 | baseline-vlm | qwen3-vl-8b-instruct | closed-set family (30-way) | orig | 902 ok / 0 err; chance=0.033 | 0 | 902 | 0.398 | 0.5843 |  |  |  |
| vlm_qwen3.7-flash_family30 | baseline-vlm | qwen3.7-flash | closed-set family (30-way) | orig | 901 ok / 1 err; chance=0.033 | 0 | 901 | 0.3896 | 0.6182 |  |  |  |
| vlm_qwen3-vl-8b-instruct | baseline-vlm | qwen3-vl-8b-instruct | open-ended | orig | 902 ok / 0 err | 0 | 902 | 0.1364 | 0.1663 |  |  |  |
| vlm_qwen3-vl-8b-instruct_no-label-leak | baseline-vlm | qwen3-vl-8b-instruct | open-ended (leak removed) | orig | 824 clean images of 902 | 0 | 824 | 0.0825 |  |  |  |  |
| vlm_qwen3.7-flash-2026-07-15 | baseline-vlm | qwen3.7-flash-2026-07-15 | open-ended | orig | 901 ok / 1 err | 0 | 901 | 0.1587 | 0.1876 |  |  |  |
| vlm_qwen3.7-flash-2026-07-15_no-label-leak | baseline-vlm | qwen3.7-flash-2026-07-15 | open-ended (leak removed) | orig | 824 clean images of 901 | 0 | 824 | 0.1092 |  |  |  |  |
| vlm_qwen3-vl-8b-instruct_species152 | baseline-vlm | qwen3-vl-8b-instruct | closed-set species (152-way) | orig | 902 ok / 3 err; chance=0.0066 | 0 | 902 | 0.2417 | 0.306 |  |  |  |
| vlm_qwen3-vl-8b-instruct_species152_no-label-leak | baseline-vlm | qwen3-vl-8b-instruct | closed-set species (152-way, leak removed) | orig | 824 clean images of 902 | 0 | 824 | 0.1833 |  |  |  |  |
| vlm_qwen3.7-flash_species152 | baseline-vlm | qwen3.7-flash | closed-set species (152-way) | orig | 901 ok / 904 err; chance=0.0066 | 0 | 901 | 0.3196 | 0.4151 |  |  |  |
| vlm_qwen3.7-flash_species152_no-label-leak | baseline-vlm | qwen3.7-flash | closed-set species (152-way, leak removed) | orig | 824 clean images of 901 | 0 | 824 | 0.2609 |  |  |  |  |

## 派生指标（`full_metrics.json`，由 `code/eval_full.py` 产出）

同一 902 张测试集上的补充指标：科级准确率（由种头派生，与 VLM 闭集科任务、
Pl@ntNet 科级数字可比）、置信校准 ECE（10 箱）、风险-覆盖率（选择性预测）、
单张推理延迟（RTX 4090 D，batch=1，预热后均值）。

| id | species_top1 | family_top1 | family_macro | ECE | latency_ms | acc@cov25% |
|---|---:|---:|---:|---:|---:|---:|
| bioclip2_ft_336 | 0.9013 | 0.9678 | 0.9617 | 0.0954 | 24.03 | 0.9911 |
| bioclip2_linear_336 | 0.9224 | 0.9712 | 0.96 | 0.1948 | 22.55 | 1.0 |
| dinov2_linear_336 | 0.8448 | 0.9324 | 0.9204 | 0.1971 | 20.97 | 0.9956 |

## 评测协议（固定不变，所有行共用）

- 数据：`finetune_expansion/dataset/` — 6,237 张 / 152 种 / 30 科；
  train 4,185 / val 882 / test 902，**按物种分层 + 按 observation 分组**，
  已验证三种划分之间 `sample_id` / `sha1` / `group_key` 交集均为 0。
- 指标：micro top-1、top-3、macro（按类平均召回），并按 `image_domain`
  （腊叶标本 / 野外照）与 `source`（GBIF / iNaturalist）分组报告。
- 开放集：700 张负样本（8 组），max-softmax 作为拒识分数，
  报告 AUROC 与 TPR@FPR5%/10%，另给分组拒识率。
- Pl@ntNet：`POST /v2/identify/all`，一次请求 = 一株植物；
  900 张直接评分，2 张内容被拒后经长边 1024 降采样恢复（记录中标记）。
- 通用 VLM：DashScope `qwen3.7-flash`（API）与本地 `Qwen3-VL-8B-Instruct`
  （vLLM 0.19.1 + torch 2.10 cu128，端口 8001，eager 模式）各跑开放式 /
  闭集科 30 选 1 / 闭集种 152 选 1 三个任务，prompt 与匹配规则逐字一致；
  泄漏校正用本地 tesseract OCR（`vlm/text_ocr.jsonl`）剔除真名可读样本。
- **迁移 / 留出类实验的协议（10-13 修正）**：标签空间在 `--train-where`
  过滤**之后**重建，测试集只保留训练时见过的物种，因此类别数 < 152 时
  不会出现零训练样本的假类别。此修正同时应用于 Batch 4（xfer，已重跑）
  与 Batch 5（geo）。
- 地理留出（Batch 5）：`country` 经 `code/clean_country.py` 归一为
  `country_clean`（ISO-2；空值/`ZZ`→UNK，占 5.5%，不参与地理分析）；
  留一国家/地区法，训练用其余全部数据，val/test 都只保留目标国家。

## 已知偏差与修正记录

| 日期 | 事项 | 处理 |
|---|---|---|
| 10-12 | Pl@ntNet 早期用前 374 张得 31.3% | **有偏**（拆分按物种排序）；全量重跑修正为 24.3%，文档已替换 |
| 10-12 | 失败记录被当作「已完成」，2 张图永不重试 | 改为仅 `ok=True` 计完成 |
| 10-12 | 终止条件在存在永久失败样本时永不成立 | 改为 `todo==0`；确定性 4xx 尝试 3 次后计入 `exhausted` |
| 10-13 | 2 张图稳定 404 | 加入降采样回退（长边 1024）并记录 `downscaled` 标记；902/902 全部评分成功 |
| 10-13 | `eval_full.py` 越界崩溃（test split 仅含 150/152 种，`ds.species` 短于训练标签空间） | 改为从 train split 重建标签空间（与 `train_fern.py` 一致）；三个 run 重跑成功 |
| 10-13 | 服务器换 IP（`<private-ip>` → `<private-ip>`），密钥免密仍有效 | 科级/校准/风险-覆盖率/延迟四项派生指标补测完成（见上表） |
| 10-13 | DashScope 充值，`species152` 闭集任务续跑完成 | n=901，top-1 0.320；剔除泄漏 0.261（`EXP_03721` 持续 HTTP400 弃置） |
| 10-13 | 新增本地 VLM 基线 Qwen3-VL-8B-Instruct（vLLM 0.19.1，端口 8001，eager） | 三任务各 902/902 全覆盖，结论与 API 版 VLM 互相印证（RESULTS.md §11） |
| 10-13 | 迁移类实验协议修正：标签空间原先取自**未过滤**的 train split，零训练样本的类别带随机权重参与竞争（field2spec 曾有多达 81/152 个假类别，gbif2inat 62 个） | `train_fern.py` 改为过滤后重建标签空间；4 个 xfer run 按新协议重跑（旧值 field2spec 0.3244 / spec2field 0.1500 / gbif2inat 0.2059 / inat2gbif 0.3239，新值见 registry），geo 批次（§12）直接采用新协议 |
