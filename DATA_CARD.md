# Data card — fern benchmark and error-budget artifacts

This card describes what is released, how it was built, what it may be used for,
and the limits a user must know before drawing conclusions from it. The paper is
the reference for method.

## Summary

| | |
|---|---|
| Name | Fern identification benchmark, 5,969 images, with shift and label-control artifacts |
| Content | Images of 152 fern species in 30 families, field photographs and herbarium specimens |
| Sources | iNaturalist research-grade observations; GBIF, including iDigBio-digitised herbarium sheets |
| Per-image annotations | image domain (field / specimen), data source (iNaturalist / GBIF), normalised country, species / genus / family under PPG I, licence and rights holder, source record ID and URL, image dimensions and SHA-1 |
| Splits | 4,185 train / 882 validation / 902 test, group-aware by observation (3,585 groups) |
| Evaluation subsets | Seven over the test split: in-domain 902, field 640, specimen 262, iNaturalist 476, GBIF 426, well-sampled species 870, sparsely sampled 32 |
| Open-set negatives | 700 images in eight groups: horsetail, lycophyte, liverwort-and-moss, cycad, palm, angiosperm, fungus, conifer |
| Label annotations | 262 boxes, one per herbarium sheet, covering the printed label and accession line (median 5.7% of the frame) |
| Redistributable images | 1,432 of 5,969 (24.0%) |
| Tasks | Closed-set classification (152 classes), open-ended naming, five-way forced choice (label control), open-set rejection |

## Composition

| Property | Balanced benchmark | Redistributable subset |
|---|---:|---:|
| Images | 5,969 | 1,432 (24.0%) |
| Species / families | 152 / 30 | — |
| Herbarium specimens | 1,669 (28.0%) | 907 (63.3%) |
| GBIF records | 2,812 (47.1%) | 1,111 (77.6%) |
| Country values after normalisation | 80 | — |
| Records without a resolvable country | 330 (5.5%) | — |

The collection total before balancing was 6,237 images; 268 were removed by a cap
of 60 images per species, drawn round-robin across the two image domains so that
capping did not collapse the domain balance. A floor of 20 images per species was
applied.

## Result files

| Folder | What it holds |
|---|---|
| `results/` | 32 runs. Every run has `config.json` and `test_metrics.json`; the eight configurations whose per-image output was retained also have `predictions.jsonl` |
| `results/jae/` | Table-level artifacts: size-matched control (per-run and summary), masked-label ablation, per-image mask results for the classifier and for the vision–language model |
| `results/analysis/` | Paired McNemar tests, per-family comparison against Pl@ntNet, accuracy summary |

Aggregate metrics are released for all 32 runs; per-image predictions only for
the eight runs that carry them. Analyses that need per-image output therefore run
on those eight, which include every configuration of the headline comparison, the
three seeds and the open-set evaluation. Transfer, geographic-holdout and
size-matched control runs are released as aggregate metrics.

## Intended use

- Evaluating closed-set accuracy, open-ended naming and open-set rejection for
  ferns, and comparing systems on a common image set.
- Measuring the cost of distribution shift along three axes that are labelled per
  image rather than inferred.
- Recomputing a domain-asymmetric accuracy profile for any published result that
  reports per-domain accuracies.
- Reusing the label-box annotation to audit text leakage in herbarium benchmarks.

## Limits

- **One taxon, one region-weighted flora.** The benchmark is fern-only and
  dominated by East Asian records; the protocol travels, the magnitudes do not.
- **The redistributable subset is not a random 24%.** It is 63.3% herbarium
  specimens against 28.0% benchmark-wide and covers 216 of the 902 test images,
  so field-side and geographic experiments cannot be reproduced from the released
  images alone.
- **Negatives are fern-like, not random.** They test rejection of look-alike taxa,
  not general out-of-distribution behaviour.
- **Label boxes bound a measurement, they do not isolate text.** A box also
  removes the accession number and barcode; the mirrored control sizes that.
- **Per-image predictions cover eight runs**, as noted above.

## Provenance and licensing

Records were harvested from iNaturalist and GBIF with the queries, date ranges
and deduplication rule documented in the paper's Supporting Information (SI-S1)
and in `data/testsets/protocol.md`. Each row keeps the licence asserted at its
source. Nothing here relicenses an image.

## Checksums

`SHA256SUMS` lists digests for the manifests, splits, testsets, label boxes and
table artifacts. Verify with `shasum -a 256 -c SHA256SUMS`.
