# Benchmark accuracy does not predict field accuracy

Code and derived data for a study of what an image classifier's reported
accuracy is worth once the conditions of deployment stop matching the conditions
of the benchmark.

The paper asks whether a held-out accuracy can stand as an estimate of field
performance, and answers it by measuring three facets of the assumption it rests
on — image domain, geographic provenance and the printed label — each against a
baseline computed on the identical images, each checked by an intervention. The
output is an **error budget**: the failure a deployment should expect,
decomposed by axis, with the data requirement that would reduce each term.

| | |
|---|---|
| Benchmark | **5,969 images**, **152 species**, **30 families** |
| Splits | 4,185 train / 882 validation / 902 test, grouped by observation |
| Open-set negatives | 700 images in eight groups (evaluation only) |
| Reference recipe | frozen BioCLIP 2 trunk + linear head at 336 px, **91.4 ± 0.7%** over three seeds |
| Measured costs | image domain or source **8.6–48.1 pp**; region **3.6–39.8 pp**; printed label **35.9 pp** (VLM) and **2.3 pp** (classifier) |
| Label annotations | 262 herbarium sheets, one box each, + mirrored control |

## What is in this repository

| Path | Contents |
|---|---|
| `data/manifests/` | `manifest_full.csv` (6,237 collected), `manifest_balanced.csv` (5,969 working set), `manifest_negatives.csv` (700), `release_manifest.csv` (1,432 redistributable) |
| `data/splits/` | Train / validation / test image lists |
| `data/testsets/` | The seven evaluation subsets and the protocol note that defines them |
| `data/label_boxes/` | The 262 annotated label-region boxes used in the masked-label control |
| `data/external_baselines/` | Per-image responses of the Pl@ntNet API, the two vision–language models and the OCR text-leakage audit |
| `data/species_data.json` | Species table: accepted names, family under PPG I, Chinese names |
| `data/figure_photos/` | The two photographs used in Fig. 1, with credits |
| `results/` | 32 runs: `config.json` and `test_metrics.json` for each; `predictions.jsonl` for the eight configurations whose per-image output was retained |
| `results/jae/` | Table-level artifacts of the paper: size-matched control, masked-label ablation, per-image mask results for both systems |
| `results/analysis/` | Paired McNemar tests, per-family comparison against Pl@ntNet, accuracy summary |
| `code/models/` | Training, full-metric evaluation, open-set evaluation, zero-shot probing |
| `code/dataset/` | Harvesting, licence classification, country normalisation, balancing, split construction, negative collection |
| `code/baselines/` | Pl@ntNet and vision–language-model harnesses, OCR audit |
| `code/analysis/` | Significance tests, confusion analysis, run summarisation |
| `code/paper_artifacts/` | `make_paper_tables.py` and `make_figures.py` — regenerate every table and figure |
| `code/scripts/` | Shell drivers for the experiment batches |
| `figures/` | The six main-text figures as vector PDFs |
| `tables/` | Source tables behind the paper's tables |
| `docs/REPRODUCE.md` | Table-by-table and figure-by-figure reproduction map |

## Reproduce the paper's numbers

Nothing below needs a GPU or the images: the scripts read the released result
files.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python code/paper_artifacts/make_paper_tables.py --out /tmp/tables
python code/paper_artifacts/make_figures.py     --out /tmp/figures
python code/analysis/paired_significance.py     --results results/runs --out /tmp/paired.csv
```

`docs/REPRODUCE.md` maps every table and figure of the paper to the file that
produces it and to the command that regenerates it.

Retraining requires the images, which are not redistributed here (see
`LICENSE-DATA.md`); `code/dataset/` rebuilds the collection from the source
records listed in the manifests, subject to each record's own licence.

## What is deliberately absent

- **The images.** 1,432 of 5,969 (24.0%) may be redistributed; the remaining
  4,537 may not. The manifests carry a source URL and licence for every row, and
  `release_manifest.csv` lists the subset that can be mirrored.
- **The manuscript source.** Author information is withheld while the paper is
  under review; the citation block is completed on acceptance.

## Licence

Code: MIT (`LICENSE`). Manifests, splits, testsets, annotations and result
files: CC BY 4.0 (`LICENSE-DATA.md`). Images keep the licence of their source
record and are not relicensed by this repository.

## Citation

While the manuscript is under review, please cite it by title:

> Benchmark accuracy does not predict field accuracy: an error budget for
> automated species identification in environmental monitoring (2026).

A `CITATION.cff` with the full author list and a DOI will be added on
acceptance. Issues and questions about the artifacts are welcome through the
issue tracker.
