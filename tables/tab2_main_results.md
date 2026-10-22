**Table 2. Main results on the 902-image test split**

Rows are single-seed (20261010) unless marked; the last three rows give the mean ± sd (population sd, ddof=0) over three seeds (20261010/11/12).

| Configuration | Mode | px | train_n | test_n | Species top-1 | top-3 | macro | specimen | field |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BioCLIP 2 / frozen / 336 | linear | 336 | 4185 | 902 | 0.9235 | 0.9778 | 0.9227 | 0.9198 | 0.9250 |
| BioCLIP 2 / fine-tuned / 336 | finetune | 336 | 4185 | 902 | 0.9013 | 0.9723 | 0.8957 | 0.9122 | 0.8969 |
| BioCLIP 2 / frozen / 224 | linear | 224 | 4185 | 902 | 0.8980 | 0.9767 | 0.8952 | 0.8969 | 0.8984 |
| BioCLIP 2 / frozen / 224 + class-bal. | linear | 224 | 4185 | 902 | 0.8980 | 0.9690 | 0.8956 | 0.8893 | 0.9016 |
| BioCLIP 2 / fine-tuned / 224 | finetune | 224 | 4185 | 902 | 0.8914 | 0.9690 | 0.8905 | 0.9046 | 0.8859 |
| BioCLIP 2 / fine-tuned / 224 (lr 5e-5) | finetune | 224 | 4185 | 902 | 0.8459 | 0.9534 | 0.8424 | 0.8855 | 0.8297 |
| BioCLIP 2 / fine-tuned / 224 + hier. head | finetune | 224 | 4185 | 902 | 0.8614 | 0.9612 | 0.8504 | 0.8779 | 0.8547 |
| DINOv2 / fine-tuned / 224 | finetune | 224 | 4185 | 902 | 0.8592 | 0.9512 | 0.8516 | 0.8702 | 0.8547 |
| DINOv2 / frozen / 336 | linear | 336 | 4185 | 902 | 0.8437 | 0.9412 | 0.8346 | 0.8626 | 0.8359 |
| DINOv2 / frozen / 224 | linear | 224 | 4185 | 902 | 0.7927 | 0.9357 | 0.7760 | 0.8015 | 0.7891 |
| BioCLIP 2 / zero-shot (no training) | zero-shot | 224 | 0 | 902 | 0.6829 | 0.8381 | 0.6864 | 0.6069 | 0.7141 |
| BioCLIP 2 / frozen / 336 — seed mean±sd | linear | 336 | 4185 | 902 | 0.9139 ± 0.0073 | — | — | — | — |
| BioCLIP 2 / fine-tuned / 336 — seed mean±sd | finetune | 336 | 4185 | 902 | 0.9084 ± 0.0041 | — | — | — | — |
| DINOv2 / frozen / 336 — seed mean±sd | linear | 336 | 4185 | 902 | 0.8433 ± 0.0050 | — | — | — | — |
