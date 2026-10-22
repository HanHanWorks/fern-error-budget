**Table 5. All systems on the same 902-image test split**

† Family accuracy derived from the species head (the VLMs instead saw an explicit 30-way family task). The last column removes images whose true name is OCR-readable on the sheet.

| System | Task | n | Species top-1 | Species top-3 | Family top-1 | top-1, leakage removed |
|---|---:|---:|---:|---:|---:|---:|
| BioCLIP 2, frozen + linear @336 (ours) | closed-set species | 902 | 0.9235 | 0.9778 | 0.971 † | — |
| DINOv2, frozen + linear @336 | closed-set species | 902 | 0.8437 | 0.9412 | 0.932 † | — |
| Pl@ntNet API (`all` referential) | open-ended | 902 | 0.2428 | 0.312 | 0.367 | — |
| Qwen3.7-flash (DashScope API) | open-ended | 901 | 0.1587 | 0.1876 | 0.522 | 0.1092 |
| Qwen3-VL-8B (local vLLM) | open-ended | 902 | 0.1364 | 0.1663 | 0.480 | 0.0825 |
| Qwen3.7-flash (DashScope API) | closed-set species | 901 | 0.3196 | 0.4151 | — | 0.2609 |
| Qwen3-VL-8B (local vLLM) | closed-set species | 902 | 0.2417 | 0.3060 | — | 0.1833 |
