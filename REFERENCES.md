# References

Everything this work builds on or compares against. Entries are taken from the paper's own
bibliography, so identifiers here match the manuscript.

**Contents:** [Cite this work](#cite-this-work) · [Event logs](#event-logs) ·
[Baselines](#baselines-we-compare-against) · [Methods](#methods-we-build-on) ·
[Licensing](#licensing-and-redistribution)

---

## Cite this work

```bibtex
@article{tran2026pfm,
  title   = {{PFM}: A Process Foundation Model with Reusable Process Representations
             for Predictive Process Monitoring},
  author  = {Tran, Trinh and W{\"o}lker, Yannick and Landsiedel, Olaf},
  journal = {Manuscript submitted to ACM},
  year    = {2026}
}
```

`CITATION.cff` carries the same metadata, so GitHub's "Cite this repository" button works.

---

## Event logs

Ten logs come from [4TU.ResearchData](https://data.4tu.nl/); [REPRODUCE.md](REPRODUCE.md#2-data)
gives the exact titles to search and the filename each must be renamed to. Two have their own
citable identifiers:

| Log | Citation |
|---|---|
| Helpdesk | Verenich, I. (2016). *Helpdesk*. [doi:10.17632/39bp3vv62t.1](https://doi.org/10.17632/39bp3vv62t.1) |
| MIMIC-IV v3.1 | Johnson, A., Bulgarelli, L., Pollard, T., Gow, B., Moody, B., Horng, S., Celi, L. A., & Mark, R. (2024). *MIMIC-IV*. PhysioNet. [doi:10.13026/kpb9-mt58](https://doi.org/10.13026/kpb9-mt58) |

The BPI Challenge logs, Road Traffic Fine Management, Hospital Billing and Sepsis Cases are
published through 4TU.ResearchData; cite the individual dataset pages there. We deliberately do not
list per-dataset DOIs we could not verify.

**If you use these logs, cite the log authors, not only this repository.**

## Baselines we compare against

Neither is vendored here. Fetch them from the authors, then use our wrappers in `hpc/submit/` and
the exported prefix manifests from `scripts/export_queries.py` so that every method is scored on
identical queries.

**SuTraN** — trained separately per target log, non-data-aware, equal-weighted configuration,
following the authors' CaLenDiR procedure.

> Wuyts, B., Vanden Broucke, S., & De Weerdt, J. (2024). SuTraN: An Encoder-Decoder Transformer for
> Full-Context-Aware Suffix Prediction of Business Processes. *ICPM 2024*, 17–24.
> [doi:10.1109/ICPM63005.2024.10680671](https://doi.org/10.1109/ICPM63005.2024.10680671)
>
> Wuyts, B., Vanden Broucke, S., & De Weerdt, J. (2025). CaLenDiR: Mitigating Case-Length Distortion
> in Deep-Learning-Based Predictive Process Monitoring. *Process Mining Workshops*, LNBIP 533,
> 253–266. [doi:10.1007/978-3-031-82225-4_19](https://doi.org/10.1007/978-3-031-82225-4_19)

**FM-v2** — the released four-expert checkpoint, loaded without updating its parameters, with `k`
selected on validation.

> Berti, A., & van der Aalst, W. M. P. (2026). An In-Context Foundation Model for Predictive Process
> Monitoring on Event Logs. *IEEE Access*, 14, 16959–16983.
> [doi:10.1109/ACCESS.2026.3658877](https://doi.org/10.1109/ACCESS.2026.3658877)
>
> Berti, A., & van der Aalst, W. M. P. (2026). Retrieval-Augmented In-Context Foundation Model for
> Predictive Process Monitoring. *Preprints*.
> [doi:10.20944/preprints202607.0705.v1](https://doi.org/10.20944/preprints202607.0705.v1)
> — preprint, not peer-reviewed.

## Methods we build on

Where each idea enters the code.

| Component | Source | Code |
|---|---|---|
| GIN message passing | Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How Powerful Are Graph Neural Networks? *ICLR* | `models/role_encoder.py` |
| Rotary position embedding | Su, J. et al. (2021). RoFormer. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) | `models/encoder.py` |
| Joint-embedding predictive architecture | Assran, M. et al. (2023). Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. *CVPR*, 15619–15629. [doi:10.1109/CVPR52729.2023.01499](https://doi.org/10.1109/CVPR52729.2023.01499) | `ssl/autoregressive.py` |
| data2vec-style latent targets | Baevski, A. et al. (2022). data2vec. *ICML*, PMLR 162, 1298–1312 | `ssl/autoregressive.py` |
| GELU | Hendrycks, D., & Gimpel, K. (2016). [arXiv:1606.08415](https://arxiv.org/abs/1606.08415) | throughout |
| PageRank | Brin, S., & Page, L. (1998). *Computer Networks and ISDN Systems*, 30(1–7), 107–117 | `data/roles.py` |
| Betweenness centrality | Freeman, L. C. (1977). *Sociometry*, 40(1), 35–41. [doi:10.2307/3033543](https://doi.org/10.2307/3033543) | `data/roles.py` |
| Entropy | Shannon, C. E. (1948). *Bell System Technical Journal*, 27(3), 379–423 | `data/roles.py` |
| Process-mining descriptors | van der Aalst, W. M. P. (2016). *Process Mining: Data Science in Action* (2nd ed.). Springer | `data/roles.py` |
| Trace-encoding choices | Tavares, G. M. et al. (2023). *Engineering Applications of AI*, 126, 107028 | `data/preprocessing.py` |
| t-SNE | van der Maaten, L., & Hinton, G. (2008). *JMLR*, 9(86), 2579–2605 | `docs/diagrams/gin15_seen_unseen.py` |
| Foundation-model framing | Bommasani, R. et al. (2021). [arXiv:2108.07258](https://arxiv.org/abs/2108.07258) | — |

Related work positioned in the paper but not used as a baseline: ProcessTransformer (Bukhsh et al.,
2021), act2vec/trace2vec (De Koninck et al., 2018), BERT-based multi-task PPM (Chen et al., 2022),
process-structure activity embeddings (Chiorrini et al., 2022), ProcessGFM (Hu et al., 2025), and
the PPM benchmarks of Rama-Maneiro et al. (2023) and Ceravolo et al. (2024).

## Licensing and redistribution

The code in this repository is under [LICENSE](LICENSE). That licence covers **the code only**.

- **Event logs are not redistributed here.** Each carries its own licence from 4TU.ResearchData or
  its publisher; check before you redistribute.
- **MIMIC-IV requires credentialed PhysioNet access** and a signed data use agreement. It cannot be
  shared, which is why `scripts/build_mimic_log.py` builds it from your own copy.
- **Baseline code is not vendored.** SuTraN and FM-v2 remain under their authors' terms.
- `results/` contains **derived metrics only** — aggregate numbers, not event data.
