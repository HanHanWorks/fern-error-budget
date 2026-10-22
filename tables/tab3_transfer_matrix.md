**Table 3. Domain / source transfer under the corrected protocol**

The label space is rebuilt from the filtered training split. 'in-domain (same subset)' is the same full model scored on the identical test subset (the convention already used in Table 4), so Δ = top-1 − in-domain isolates the transfer cost from subset composition. The last column keeps the values published before the label-space correction.

| Experiment | Train on | Test on | Classes | train_n | test_n | top-1 | macro | in-domain (same subset) | Δ (pp) | previous value |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cross-source | GBIF | iNaturalist | 90 | 1965 | 128 | 0.7891 | 0.8071 | 0.8750 | -8.6 | 0.2059 (defective) |
| Cross-domain | specimen | field | 71 | 1166 | 156 | 0.6538 | 0.7260 | 0.8526 | -19.9 | 0.3244 (defective) |
| Cross-source | iNaturalist | GBIF | 122 | 2220 | 270 | 0.5000 | 0.5697 | 0.9000 | -40.0 | 0.3239 (defective) |
| Cross-domain | field | specimen | 137 | 3019 | 210 | 0.4238 | 0.4285 | 0.9048 | -48.1 | 0.1500 (defective) |
| In-domain | all | all | 152 | 4185 | 902 | 0.9235 | 0.9227 | 0.9235 | 0.0 | — |
