**Table 6. Paired McNemar tests on the shared test images**

All systems saw the same images; *** p<0.001, ** p<0.01, * p<0.05, n.s. = not significant. Full 105-pair matrix in paired_mcnemar_v2.csv. **All tests are single-seed (20261010):** the bioclip2_linear_336 vs bioclip2_ft_336 row (+2.1 pp, p=0.025) does NOT survive seed variance -- across three seeds the paired difference is +2.00 / +0.44 / -0.78 pp (mean +0.55 ± 1.13 pp, interval includes zero), so frozen and fine-tuned are statistically indistinguishable. Cite the seed-level result (Table 2 and RESULTS.md §14), not this single-seed p-value.

| Comparison | n (paired) | Δ top-1 (pp) | McNemar p | sig. | 95% CI of Δ |
|---|---:|---:|---:|---:|---:|
| bioclip2_linear_336 vs dinov2_linear_336 | 902 | +7.8 | 0.00e+00 | *** | [+5.5, +10.1] |
| bioclip2_linear_336 vs bioclip2_ft_336 | 902 | +2.1 | 2.48e-02 | * | [+0.4, +3.9] |
| bioclip2_linear_336 vs bioclip2_ft_hier | 902 | +6.0 | 0.00e+00 | *** | [+4.0, +8.1] |
| bioclip2_linear_336 vs bioclip2_linear | 902 | +2.5 | 5.15e-03 | ** | [+0.9, +4.3] |
| bioclip2_linear_336 vs plantnet_all | 902 | +68.0 | 0.00e+00 | *** | [+64.8, +71.1] |
| bioclip2_linear_336 vs qwen3.7-flash_species152 | 901 | +60.3 | 0.00e+00 | *** | [+56.8, +63.6] |
| bioclip2_linear_336 vs qwen3-vl-8b_species152 | 902 | +68.1 | 0.00e+00 | *** | [+64.9, +71.2] |
| bioclip2_linear_336 vs qwen3.7-flash_open_noleak | 824 | +80.8 | 0.00e+00 | *** | [+78.0, +83.5] |
| bioclip2_linear_336 vs qwen3-vl-8b_open_noleak | 824 | +83.5 | 0.00e+00 | *** | [+80.8, +86.0] |
| dinov2_linear_336 vs dinov2_ft | 902 | -1.4 | 2.67e-01 | n.s. | [-3.8, +0.9] |
| plantnet_all vs qwen3.7-flash_open | 901 | +8.4 | 1.00e-05 | *** | [+4.8, +12.2] |
