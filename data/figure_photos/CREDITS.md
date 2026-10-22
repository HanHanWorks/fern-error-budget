# Photograph credits (Figure 1)

The two photographs embedded in `manuscript/figures/Figure1_workflow.png`
show the same species, *Cyrtomium macrophyllum* (Dryopteridaceae), in the two
image regimes the benchmark distinguishes. Both are redistributed under the
licence recorded for each image in `data/manifests/manifest_balanced.csv`.

| File | Regime | Licence | Rights holder | Source record |
|---|---|---|---|---|
| `EXP_01736.jpg` | herbarium sheet | CC0 | none required (PRESERVED_SPECIMEN, China) | GBIF `1322235008`, US National Herbarium, Smithsonian Institution — <http://n2t.net/ark:/65665/3de9e2058-2024-4c6a-9399-d7590079586c> |
| `EXP_01734.jpg` | field photograph | CC BY 4.0 | Cynthia Su (HUMAN_OBSERVATION, Taiwan, China) | iNaturalist observation 13934276, GBIF `1891291413` — <https://www.inaturalist.org/observations/13934276> |

CC0 requires no attribution; the herbarium sheet is credited anyway as good
practice. The CC BY photograph must be credited to Cynthia Su whenever the
figure is reused; the manuscript's Figure 1 caption carries this attribution.

Regenerating Figure 1 requires both files in this directory
(`tools/stage_release.py` places them here); `code/paper_artifacts/make_figures.py`
resolves them by sample identifier and re-checks the licence before embedding.
