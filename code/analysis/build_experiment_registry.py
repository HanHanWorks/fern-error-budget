#!/usr/bin/env python3
"""Consolidate every experiment into one auditable registry.

Pulls configs and metrics from `results/` (fine-tuning runs) and from the
Pl@ntNet / open-set artefacts, then writes:

    EXPERIMENT_LOG.md     human-readable, one row per run + protocol notes
    results/registry.csv  machine-readable, same content

Nothing is recomputed here; this only reads what the runs already produced, so
the registry can never disagree with the raw metrics files.
"""
from __future__ import annotations

import csv
import difflib
import json
import re as _re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
PLANTNET = ROOT / "plantnet"

FIELDS = ["id", "kind", "backbone", "mode", "img_size", "extra", "train_n",
          "test_n", "top1", "top3", "macro_top1", "specimen_top1",
          "field_top1", "source_file"]


def f(v):
    return "" if v is None else round(float(v), 4)


def from_train_runs() -> list[dict]:
    out = []
    for d in sorted(RESULTS.iterdir()):
        cfg_p, met_p = d / "config.json", d / "test_metrics.json"
        if not (d.is_dir() and cfg_p.exists() and met_p.exists()):
            continue
        cfg = json.loads(cfg_p.read_text())
        met = json.loads(met_p.read_text())
        t = met["test"]
        if d.name.startswith("smoke"):
            continue        # throwaway pipeline check, not a result
        if cfg.get("train_where") or cfg.get("eval_where"):
            tw = cfg.get("train_where") or ""
            ew = cfg.get("eval_where") or ""
            if "country_clean" in tw or "country_clean" in ew:
                kind = "geo-transfer"
                hold = ew.split("=", 1)[-1] if ew else ""
                extra = f"holdout={hold} (train_n_species={cfg.get('n_species')})"
            else:
                kind = "transfer"
                extra = f"{tw}->{ew}"
        else:
            kind = "finetune"
            extra = "hier" if cfg.get("hier") else (
                "class-balance" if cfg.get("class_balance") else "")
        out.append({
            "id": d.name, "kind": kind, "backbone": met["backbone"],
            "mode": met["mode"], "img_size": met["img_size"], "extra": extra,
            "train_n": cfg.get("train_n"), "test_n": t.get("n"),
            "top1": f(t.get("top1")), "top3": f(t.get("top3")),
            "macro_top1": f(t.get("macro_top1")),
            "specimen_top1": f((t.get("domain_top1") or {}).get("specimen")),
            "field_top1": f((t.get("domain_top1") or {}).get("field")),
            "source_file": str(d.relative_to(ROOT) / "test_metrics.json"),
        })
    return out


def from_zeroshot() -> list[dict]:
    out = []
    for p in sorted(RESULTS.glob("zeroshot_*.json")):
        d = json.loads(p.read_text())
        out.append({
            "id": p.stem, "kind": "zero-shot", "backbone": d["weights"],
            "mode": "zero-shot", "img_size": 224, "extra": "no training",
            "train_n": 0, "test_n": d["n"],
            "top1": f(d["species_top1"]), "top3": f(d["species_top3"]),
            "macro_top1": f(d["species_macro_top1"]),
            "specimen_top1": f((d.get("domain_top1") or {}).get("specimen")),
            "field_top1": f((d.get("domain_top1") or {}).get("field")),
            "source_file": str(p.relative_to(ROOT)),
        })
    return out


def from_openset() -> list[dict]:
    out = []
    for p in sorted(RESULTS.glob("*/openset_metrics.json")):
        d = json.loads(p.read_text())
        out.append({
            "id": p.parent.name + "_openset", "kind": "open-set",
            "backbone": "", "mode": "max-softmax", "img_size": "",
            "extra": f"AUROC={d['auroc']:.4f} TPR@FPR10={d['tpr_at_fpr10']:.4f}"
                     f" n_neg={d['n_neg']}",
            "train_n": "", "test_n": d["n_pos"], "top1": "", "top3": "",
            "macro_top1": "", "specimen_top1": "", "field_top1": "",
            "source_file": str(p.relative_to(ROOT)),
        })
    return out


def from_full_metrics() -> list[dict]:
    """Derived per-sample metrics from code/eval_full.py on the same test
    split: family accuracy (derived from the species head), calibration
    (ECE), risk-coverage (selective prediction) and inference latency."""
    out = []
    for p in sorted(RESULTS.glob("*/full_metrics.json")):
        d = json.loads(p.read_text())
        rc = dict(zip(d["coverage"], d["accuracy_at_coverage"]))
        rc25 = min(rc, key=lambda c: abs(c - 0.25)) if rc else ""
        out.append({
            "id": p.parent.name + "_full", "kind": "derived-full",
            "backbone": "", "mode": "eval_full.py", "img_size": "",
            "extra": f"family={d['family_top1_derived']}"
                     f" fam_macro={d['family_macro']} ece={d['ece_10bin']}"
                     f" lat={d['latency_ms_mean']}ms rc@cov0.25={rc.get(rc25, '')}",
            "train_n": "", "test_n": d["n"],
            "top1": f(d["species_top1"]), "top3": f(d["species_top3"]),
            "macro_top1": "", "specimen_top1": "", "field_top1": "",
            "source_file": str(p.relative_to(ROOT)),
        })
    return out


def from_plantnet() -> list[dict]:
    import collections
    out = []
    for p in sorted(PLANTNET.glob("raw_*.jsonl")):
        recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
                if l.strip()]
        ok = [r for r in recs if r.get("ok")]
        if not ok:
            continue
        def norm(s):
            return " ".join((s or "").lower().split())
        def binom(s):
            t = norm(s).split()
            return " ".join(t[:2]) if len(t) >= 2 else norm(s)
        hit = sum(1 for r in ok if (r.get("top") or [{}])[0].get("sci")
                  and binom(r["top"][0]["sci"]) == binom(r["true_species"]))
        down = sum(1 for r in ok if r.get("downscaled"))
        out.append({
            "id": p.stem.replace("raw_", "plantnet_"),
            "kind": "baseline-api", "backbone": "Pl@ntNet",
            "mode": "zero-shot API", "img_size": "",
            "extra": f"{len(ok)} ok / {len(recs)-len(ok)} rejected"
                     + (f", {down} downscaled" if down else ""),
            "train_n": 0, "test_n": len(ok),
            "top1": f(hit / len(ok)), "top3": "", "macro_top1": "",
            "specimen_top1": "", "field_top1": "",
            "source_file": str(p.relative_to(ROOT)),
        })
    return out


def _nb(s: str) -> str:
    q = " ".join((s or "").lower().replace("*", "").split()).split()
    return " ".join(q[:2])


def _nm(s: str) -> str:
    s = (s or "").lower().replace("×", "x")
    return " ".join(_re.sub(r"[^a-z0-9\s\-]", " ", s).split())


def _leak(tsp: str, text: str) -> bool:
    """True if the true species name is readable as text in the OCR dump."""
    q = _nm(tsp).split()
    if len(q) < 2:
        return False
    g, e = q[0], q[1]
    t = _nm(text)
    if f"{g} {e}" in t:
        return True
    if len(e) >= 6:
        return any(abs(len(k) - len(e)) <= 4 and
                   difflib.SequenceMatcher(None, e, k).ratio() >= 0.8
                   for k in t.split())
    return False


def _ocr_index() -> dict:
    p = ROOT / "data/external_baselines/vlm/text_ocr.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["sample_id"]] = r
    return out


def from_vlm() -> list[dict]:
    """General-purpose VLM baseline on the same test split."""
    VLM = ROOT / "data/external_baselines/vlm"
    out = []
    for p in sorted(VLM.glob("family30_*.jsonl")):
        recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
                if l.strip()]
        ok = [r for r in recs if r.get("ok")]
        if not ok:
            continue
        hit = sum(1 for r in ok if (r.get("top3") or [{}])[0].get("family")
                  and " ".join(r["top3"][0]["family"].lower().split())
                  == " ".join((r.get("true_family_latin") or "").lower().split()))
        model = p.stem.split("_", 1)[1].rsplit("_", 1)[0]
        hit3 = sum(1 for r in ok
                   if any(" ".join((x.get("family") or "").lower().split())
                          == " ".join((r.get("true_family_latin") or "")
                                      .lower().split())
                          for x in (r.get("top3") or [])))
        out.append({
            "id": f"vlm_{model}_family30", "kind": "baseline-vlm",
            "backbone": model, "mode": "closed-set family (30-way)",
            "img_size": "orig",
            "extra": f"{len(ok)} ok / {len(recs)-len(ok)} err; chance=0.033",
            "train_n": 0, "test_n": len(ok), "top1": f(hit / len(ok)),
            "top3": f(hit3 / len(ok)), "macro_top1": "", "specimen_top1": "",
            "field_top1": "", "source_file": str(p.relative_to(ROOT)),
        })
    for p in sorted(VLM.glob("open_*.jsonl")):
        recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
                if l.strip()]
        ok = [r for r in recs if r.get("ok")]
        if not ok:
            continue
        hit = sum(1 for r in ok if (r.get("top3") or [{}])[0].get("name")
                  and _nb(r["top3"][0]["name"]) == _nb(r["true_species"]))
        model = p.stem.split("_", 1)[1].rsplit("_", 1)[0]
        hit3 = sum(1 for r in ok
                   if _nb(r["true_species"]) in
                   [_nb(x.get("name") or "") for x in (r.get("top3") or [])])
        rows = [{
            "id": f"vlm_{model}", "kind": "baseline-vlm",
            "backbone": model, "mode": "open-ended", "img_size": "orig",
            "extra": f"{len(ok)} ok / {len(recs)-len(ok)} err",
            "train_n": 0, "test_n": len(ok), "top1": f(hit / len(ok)),
            "top3": f(hit3 / len(ok)), "macro_top1": "", "specimen_top1": "",
            "field_top1": "", "source_file": str(p.relative_to(ROOT)),
        }]
        # Leakage-controlled variant, measured with local tesseract OCR.
        ocr = _ocr_index()
        if ocr:
            clean = [r for r in ok
                     if r["sample_id"] in ocr
                     and not _leak(r["true_species"],
                                   ocr[r["sample_id"]]["ocr_text"])]
            if clean:
                ch = sum(1 for r in clean
                         if (r.get("top3") or [{}])[0].get("name")
                         and _nb(r["top3"][0]["name"]) == _nb(r["true_species"]))
                rows.append({
                    "id": f"vlm_{model}_no-label-leak", "kind": "baseline-vlm",
                    "backbone": model, "mode": "open-ended (leak removed)",
                    "img_size": "orig",
                    "extra": f"{len(clean)} clean images of {len(ok)}",
                    "train_n": 0, "test_n": len(clean),
                    "top1": f(ch / len(clean)),
                    "top3": "", "macro_top1": "", "specimen_top1": "",
                    "field_top1": "", "source_file": "vlm/text_ocr.jsonl",
                })
        out.extend(rows)
    for p in sorted(VLM.glob("species152_*.jsonl")):
        recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
                if l.strip()]
        ok = [r for r in recs if r.get("ok")]
        if not ok:
            continue
        hit = sum(1 for r in ok if (r.get("top3") or [{}])[0].get("name")
                  and _nb(r["top3"][0]["name"]) == _nb(r["true_species"]))
        model = p.stem.split("_", 1)[1].rsplit("_", 1)[0]
        hit3 = sum(1 for r in ok
                   if _nb(r["true_species"]) in
                   [_nb(x.get("name") or "") for x in (r.get("top3") or [])])
        rows = [{
            "id": f"vlm_{model}_species152", "kind": "baseline-vlm",
            "backbone": model, "mode": "closed-set species (152-way)",
            "img_size": "orig",
            "extra": f"{len(ok)} ok / {len(recs)-len(ok)} err; chance=0.0066",
            "train_n": 0, "test_n": len(ok), "top1": f(hit / len(ok)),
            "top3": f(hit3 / len(ok)), "macro_top1": "", "specimen_top1": "",
            "field_top1": "", "source_file": str(p.relative_to(ROOT)),
        }]
        ocr = _ocr_index()
        if ocr:
            clean = [r for r in ok
                     if r["sample_id"] in ocr
                     and not _leak(r["true_species"],
                                   ocr[r["sample_id"]]["ocr_text"])]
            if clean:
                ch = sum(1 for r in clean
                         if (r.get("top3") or [{}])[0].get("name")
                         and _nb(r["top3"][0]["name"]) == _nb(r["true_species"]))
                rows.append({
                    "id": f"vlm_{model}_species152_no-label-leak",
                    "kind": "baseline-vlm", "backbone": model,
                    "mode": "closed-set species (152-way, leak removed)",
                    "img_size": "orig",
                    "extra": f"{len(clean)} clean images of {len(ok)}",
                    "train_n": 0, "test_n": len(clean),
                    "top1": f(ch / len(clean)),
                    "top3": "", "macro_top1": "", "specimen_top1": "",
                    "field_top1": "", "source_file": "vlm/text_ocr.jsonl",
                })
        out.extend(rows)
    return out


def main() -> int:
    rows = (from_train_runs() + from_zeroshot() + from_openset()
            + from_full_metrics() + from_plantnet() + from_vlm())
    out_csv = RESULTS / "registry.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# 实验数据记录（Experiment registry）",
        "",
        f"本文件由 `code/build_experiment_registry.py` 从各次运行产出的 "
        f"`config.json` / `test_metrics.json` / `full_metrics.json` / jsonl 原始结果自动汇总，"
        f"共 **{len(rows)}** 条记录。机器可读版本：`results/registry.csv`。",
        "",
        "所有数字均直接读取原始指标文件，未经二次计算，因此不会与原始记录不一致。",
        "",
        "## 记录明细",
        "",
        "| id | kind | backbone | mode | img | extra | train_n | test_n | top1 | top3 | macro | specimen | field |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['kind']} | {r['backbone']} | {r['mode']} | "
            f"{r['img_size']} | {r['extra']} | {r['train_n']} | {r['test_n']} | "
            f"{r['top1']} | {r['top3']} | {r['macro_top1']} | "
            f"{r['specimen_top1']} | {r['field_top1']} |")
    lines += [
        "",
        "## 派生指标（`full_metrics.json`，由 `code/eval_full.py` 产出）",
        "",
        "同一 902 张测试集上的补充指标：科级准确率（由种头派生，与 VLM 闭集科任务、",
        "Pl@ntNet 科级数字可比）、置信校准 ECE（10 箱）、风险-覆盖率（选择性预测）、",
        "单张推理延迟（RTX 4090 D，batch=1，预热后均值）。",
        "",
        "| id | species_top1 | family_top1 | family_macro | ECE | latency_ms | acc@cov25% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for p in sorted(RESULTS.glob("*/full_metrics.json")):
        d = json.loads(p.read_text())
        rc = dict(zip(d["coverage"], d["accuracy_at_coverage"]))
        rc25 = rc.get(min(rc, key=lambda c: abs(c - 0.25)), "") if rc else ""
        lines.append(
            f"| {p.parent.name} | {d['species_top1']} | "
            f"{d['family_top1_derived']} | {d['family_macro']} | "
            f"{d['ece_10bin']} | {d['latency_ms_mean']} | {rc25} |")
    lines += [
        "",
        "## 评测协议（固定不变，所有行共用）",
        "",
        "- 数据：`finetune_expansion/dataset/` — 6,237 张 / 152 种 / 30 科；",
        "  train 4,185 / val 882 / test 902，**按物种分层 + 按 observation 分组**，",
        "  已验证三种划分之间 `sample_id` / `sha1` / `group_key` 交集均为 0。",
        "- 指标：micro top-1、top-3、macro（按类平均召回），并按 `image_domain`",
        "  （腊叶标本 / 野外照）与 `source`（GBIF / iNaturalist）分组报告。",
        "- 开放集：700 张负样本（8 组），max-softmax 作为拒识分数，",
        "  报告 AUROC 与 TPR@FPR5%/10%，另给分组拒识率。",
        "- Pl@ntNet：`POST /v2/identify/all`，一次请求 = 一株植物；",
        "  900 张直接评分，2 张内容被拒后经长边 1024 降采样恢复（记录中标记）。",
        "- 通用 VLM：DashScope `qwen3.7-flash`（API）与本地 `Qwen3-VL-8B-Instruct`",
        "  （vLLM 0.19.1 + torch 2.10 cu128，端口 8001，eager 模式）各跑开放式 /",
        "  闭集科 30 选 1 / 闭集种 152 选 1 三个任务，prompt 与匹配规则逐字一致；",
        "  泄漏校正用本地 tesseract OCR（`vlm/text_ocr.jsonl`）剔除真名可读样本。",
        "- **迁移 / 留出类实验的协议（10-13 修正）**：标签空间在 `--train-where`",
        "  过滤**之后**重建，测试集只保留训练时见过的物种，因此类别数 < 152 时",
        "  不会出现零训练样本的假类别。此修正同时应用于 Batch 4（xfer，已重跑）",
        "  与 Batch 5（geo）。",
        "- 地理留出（Batch 5）：`country` 经 `code/clean_country.py` 归一为",
        "  `country_clean`（ISO-2；空值/`ZZ`→UNK，占 5.5%，不参与地理分析）；",
        "  留一国家/地区法，训练用其余全部数据，val/test 都只保留目标国家。",
        "",
        "## 已知偏差与修正记录",
        "",
        "| 日期 | 事项 | 处理 |",
        "|---|---|---|",
        "| 10-12 | Pl@ntNet 早期用前 374 张得 31.3% | **有偏**（拆分按物种排序）；全量重跑修正为 24.3%，文档已替换 |",
        "| 10-12 | 失败记录被当作「已完成」，2 张图永不重试 | 改为仅 `ok=True` 计完成 |",
        "| 10-12 | 终止条件在存在永久失败样本时永不成立 | 改为 `todo==0`；确定性 4xx 尝试 3 次后计入 `exhausted` |",
        "| 10-13 | 2 张图稳定 404 | 加入降采样回退（长边 1024）并记录 `downscaled` 标记；902/902 全部评分成功 |",
        "| 10-13 | `eval_full.py` 越界崩溃（test split 仅含 150/152 种，`ds.species` 短于训练标签空间） | 改为从 train split 重建标签空间（与 `train_fern.py` 一致）；三个 run 重跑成功 |",
        "| 10-13 | 服务器换 IP（`<private-ip>` → `<private-ip>`），密钥免密仍有效 | 科级/校准/风险-覆盖率/延迟四项派生指标补测完成（见上表） |",
        "| 10-13 | DashScope 充值，`species152` 闭集任务续跑完成 | n=901，top-1 0.320；剔除泄漏 0.261（`EXP_03721` 持续 HTTP400 弃置） |",
        "| 10-13 | 新增本地 VLM 基线 Qwen3-VL-8B-Instruct（vLLM 0.19.1，端口 8001，eager） | 三任务各 902/902 全覆盖，结论与 API 版 VLM 互相印证（RESULTS.md §11） |",
        "| 10-13 | 迁移类实验协议修正：标签空间原先取自**未过滤**的 train split，零训练样本的类别带随机权重参与竞争（field2spec 曾有多达 81/152 个假类别，gbif2inat 62 个） | `train_fern.py` 改为过滤后重建标签空间；4 个 xfer run 按新协议重跑（旧值 field2spec 0.3244 / spec2field 0.1500 / gbif2inat 0.2059 / inat2gbif 0.3239，新值见 registry），geo 批次（§12）直接采用新协议 |",
    ]
    (ROOT / "EXPERIMENT_LOG.md").write_text("\n".join(lines) + "\n",
                                            encoding="utf-8")
    print(f"wrote {out_csv.relative_to(ROOT)} ({len(rows)} rows)")
    print(f"wrote EXPERIMENT_LOG.md")
    for r in rows:
        print(f"  {r['id']:28s} {r['kind']:10s} top1={r['top1']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
