**Table 8. Species-level coverage and error structure**

'Species represented' counts species with ≥2 test images (148). The 902 test images cover 150 species in total; *Huperzia nanchuanensis* and *H. sutchueniana* contribute a single image each and are excluded only from the species-level counts, not from accuracy. 'Never in the top-3' is computed over each system's ranked candidate list (rank ≤ 3). Family errors are counted against the benchmark's PPG I family labels, using Pl@ntNet's own family field for its predictions.

| Metric | BioCLIP 2 linear@336 | Pl@ntNet |
|---|---:|---:|
| Test images evaluated | 902 | 902 |
| Species represented (n ≥ 2 test images) | 148 | 148 |
| Never identified correctly | 0 (0.0%) | 90 (60.8%) |
| Never in the top-3 | 0 (0.0%) | 86 (58.1%) |
| Cross-genus errors | 47 (5.2%) | 446 (49.4%) |
| Cross-family errors | 26 (2.9%) | 473 (52.4%) |
| Largest single confusion direction | 3 images | 188 of 273 Dryopteridaceae images → Polypodiaceae |
