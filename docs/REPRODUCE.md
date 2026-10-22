# Reproduction map

Every number in the paper traces to a file in this repository. Tables 1–4 and
Figures 3–6 regenerate from released artifacts without a GPU or the images;
Table 5 follows the procedure in SI-S6 and needs the per-image predictions that
are released for the frozen reference; Table 6 is qualitative and has no
artifact; Figures 1–2 are schematics, shipped as the PDFs used in the manuscript.

## Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`make_paper_tables.py` needs only the standard library. `make_figures.py` needs
matplotlib. Nothing here loads a model.

## Commands

```bash
python code/paper_artifacts/make_paper_tables.py --out out/tables
python code/paper_artifacts/make_figures.py      --out out/figures
python code/analysis/paired_significance.py      --boot 10000
```

`make_paper_tables.py` reads the manifests, the splits, `results/registry.csv`
and every `results/<run>/test_metrics.json`; it writes each table as both CSV and
Markdown. `make_figures.py` reads the same result files and writes Figures 3–8 as
PNG at 300 dpi.

## Tables

| Paper | Contents | Produced by | File |
|---|---|---|---|
| Table 1 | Benchmark composition | `make_paper_tables.py` | `out/tables/tab1_dataset_composition.csv` |
| Table 2 | Domain and source transfer, full budget | `make_paper_tables.py` | `out/tables/tab3_transfer_matrix.csv` |
| Table 2 (lower block) | Size-matched control, 1,166 images | released artifact | `results/jae/table3_size_matched_runs.csv`, `..._summary.csv` |
| Table 3 | Leave-one-region-out holdout | `make_paper_tables.py` | `out/tables/tab4_geographic_holdout.csv` |
| Table 4 | Masked-label control, three conditions | released artifact | `results/jae/table7_mask_ablation.csv` |
| Table 4 (per image) | Per-sheet scores in three conditions | released artifact | `results/jae/table7_mask_per_image_bioclip2.jsonl`, `..._qwen3vl8b.jsonl` |
| Table 5 | Acceptance-test sizing | procedure in SI-S6, applied to `results/bioclip2_linear_336/predictions.jsonl` | no dedicated script |
| Table 6 | Mechanism predictions and falsifiers | qualitative | — |

## Figures

| Paper | Contents | Produced by |
|---|---|---|
| Fig. 1 | Evaluation framework | schematic; `figures/Figure1_workflow.pdf` is the vector file used |
| Fig. 2 | Benchmark construction | schematic; `figures/Figure2_pipeline.pdf` |
| Fig. 3 | Configuration benchmark | `make_figures.py` → `Figure3_main_results.png` |
| Fig. 4 | Transfer cost by direction | `make_figures.py` → `Figure4_transfer_drops.png` |
| Fig. 5 | Geographic holdout | `make_figures.py` → `Figure5_geographic_holdout.png` |
| Fig. 6 | Masked-label ablation | `make_figures.py` → `Figure_mask_ablation` equivalent; values from `results/jae/table7_mask_ablation.csv` |
| SI Figs S1–S3 | Reliability, risk–coverage, error structure | `make_figures.py` → `Figure6_reliability.png`, `Figure7_risk_coverage.png`, `Figure8_error_structure.png` |

## Per-image predictions

`results/<run>/predictions.jsonl` exists for the eight configurations whose
per-image output was retained:

```
bioclip2_linear_336   bioclip2_ft_336   bioclip2_linear   bioclip2_ft
dinov2_linear         dinov2_linear_336 dinov2_ft         bioclip2_ft_hier
```

Analyses that need per image output (selective prediction, acceptance-test
sizing, error structure) run on these. Transfer, geographic-holdout and
size-matched control runs are released as aggregate metrics in
`results/<run>/test_metrics.json`.

## Known limits of the release

- The redistributable images cover 216 of the 902 test images, 63.3% of them
  herbarium sheets; field-side and geographic experiments cannot be re-run from
  the released images alone (`data/manifests/release_manifest.csv`).
- Table 5 is a resampling calculation, not a stored table; SI-S6 gives the
  procedure and the input is the released prediction file.
- Figure 1 and Figure 2 were drawn in PowerPoint; the PDFs are the archival
  form and the layout is not script-generated.
