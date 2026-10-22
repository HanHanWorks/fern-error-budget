# Evaluation protocol

All subsets are derived from `dataset/splits/test.csv`; no image is
moved or removed. Group-aware splitting guarantees that every image of
an observation stays inside one split.

| subset | purpose |
|---|---|
| test_indomain | headline number: mixed-domain, mixed-source |
| test_field | field photographs only |
| test_specimen | herbarium sheets only |
| test_inaturalist | citizen-science photographs |
| test_gbif | museum / aggregate records |
| test_core | species with >= 30 balanced images |
| test_rare | species with < 30 balanced images |

## Why this protocol

Training on the full train split and testing on the full test split
reports 92.4% top-1. Training and testing *within* a single domain or
source and crossing them collapses to 15-32%. Reporting only the mixed
number therefore overstates readiness for deployment, where the image
domain of the field stream rarely matches the training corpus.

## Transfer matrix to report

| train | test | subset files |
|---|---|---|
| source=GBIF | source=iNaturalist | train.csv / test_inaturalist.csv |
| source=iNaturalist | source=GBIF | train.csv / test_gbif.csv |
| image_domain=specimen | image_domain=field | train.csv / test_field.csv |
| image_domain=field | image_domain=specimen | train.csv / test_specimen.csv |

## Known limitations

- Two species (*Neolepisorus fortunei*, *Huperzia nanchuanensis*) have too
  few images to appear in every split.
- `country_iso` was recovered heuristically from a free-text field; a
  geography holdout needs manual verification before it is claimed.
- No open-set negatives are included; rejection cannot be evaluated yet.

## Country coverage in the test split (44 ISO codes recovered)

| code | images |
|---|---:|
| TW | 195 |
| CN | 150 |
| JP | 57 |
| RU | 33 |
| US | 31 |
| TH | 27 |
| VN | 26 |
| NP | 18 |
| IN | 18 |
| MY | 17 |
| KR | 10 |
| ID | 9 |
| ZZ | 8 |
| HK | 5 |
| CV | 4 |
