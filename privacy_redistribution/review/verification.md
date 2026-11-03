# Verification: iteration 2 (fixes for review.json findings 1–4)

Paper: `/Users/olukotunjosh/Downloads/VSCA/privacy_redistribution/paper/iclr2025_conference.tex` + `iclr2025_conference.bib`.
I did not edit `backup_original/`. All 8 files in it are still byte-identical to `proj.zip` (checked with `cmp`).
I did not upload anything to Overleaf. The `.bib` did not change in this iteration.

## 0. Changes made for each review finding

| Finding | Change in the .tex |
|---|---|
| 1. Δ^del mixed M=32 and M_B=16 attack configurations; the variance rule was ambiguous | §3.10: a separate **no-deletion baseline set** of M_B models per Study B condition (CE, True-Hier) on random halves of D_β. It has its own membership matrix and disjoint seeds, and it is scored with the same leave-one-model-out procedure, variance rule and α as the deletion sets. Eq. 21 (Δ^del) now says the second term comes from this set and not from the Study A set. The text says the remaining pool-size effect is shared by every removal set of size \|R\| and is controlled by R^rand, so a positive value is read *relative to R^rand*. §3.5: the variance rule now uses the per-target reference counts. Per-record variances are used only if \|IN_i∖{m}\| ≥ 15 and \|OUT_i∖{m}\| ≥ 15 for every target. That gives per-record variances at M=32 and pooled variances at 16. Table 3 has a new row (2 × 16 = 32 runs, 11 GPU-h), so the totals are now 949 runs and 326 GPU-h. The "about 326 A100-hours" figure in §4.2 was updated to match. |
| 2. ΔU is degenerate on the primary removal sets; the draft's ΔV-matching clause was dropped | §3.9: new **sibling support mass** K_i^G (Eq. 18), the unnormalized sum of κ over siblings, which responds to uniform removal. Visual-neighbor density V_i is now a defined equation (Eq. 19). U (Eq. 17) is unchanged, and the text states that E[ΔU]=0 under uniform within-class removal. §3.10 "Support change": ΔK, ΔN and ΔV are defined. ΔK^{G_true} is the primary support-change quantity: negative under R^sem, exactly 0 for cohort records under R^vis, small under R^rand. New paragraph "**ΔV matching**": R^vis is selected by a same-class swap search so that its cohort-mean ΔV matches that of R^sem (5% tolerance, at most 10^4 swaps, uses only features). This implements the source draft's §2.10.2 clause. The R^vis bullet points to this step. §3.11: X^sem (Eq. 26) adds K and ΔK and keeps N, ΔN, U, ΔU. §3.12: the linking pair is now (ΔK^{G_true}, Δ^del). Table 2 "Support features" and "Deletion sets" rows were updated. |
| 3. R^hv and r_i^0 were not well defined | §3.10: new "**Preliminary audit**" paragraph and Eq. 22. These define the preliminary score Λ̄_i^0 (mean leave-one-model-out Λ over IN models on M^0), r_i^0 = r_i(CE, D_β; α=0.1%) and Mem_i^0 = Mem_i(CE, D_β), all on M^0. R^hv takes the top \|R\| records of D_β∖T_q by the continuous Λ̄_i^0, ties broken by smaller record index. At α=0.1% with M_0=16 there are 320,000 nonmember pairs, so about 320 expected false positives, which satisfies the ≥100 rule. |
| 4. RQ4 dropped "influence-related measures" | §1 RQ4 wording restored: "…baseline difficulty and influence-related measures (baseline memorization and baseline vulnerability)". X^base (Eq. 25) adds Mem_i^0 (the influence-related control) and Λ̄_i^0. |

The rest of the paper is unchanged.

## 1. Compile

Command, run in `paper/`:

```
/opt/homebrew/bin/tectonic -X compile --keep-logs iclr2025_conference.tex
```

Result: exit 0. The PDF has 14 pages.

- Errors: none.
- Undefined references: none. Undefined citations: none. The log has no `undefined` lines, and the BibTeX `.blg` has 0 warnings.
- Overfull boxes: **none**. The log has no `Overfull` lines.
- Remaining warning: one `Underfull \vbox (badness 10000) has occurred while \output is active`. It was already there in iteration 1 and comes from a float page break. No text runs outside the margins.
- Eq. 26 (X^sem) first rendered too wide, which pushed its equation number onto its own line. I set it as a two-line `aligned`, and it now fits.

Length: the body runs to about page 11. It grew by about half a page because of the new paragraphs. References start on page 11, Appendix A/B is on pages 13–14, and the last 3 lines of Appendix B spill onto page 14.

## 2. Rendered pages

Rendered with Swift + PDFKit (`swift /tmp/render.swift paper/iclr2025_conference.pdf review/pages`) to
`/Users/olukotunjosh/Downloads/VSCA/privacy_redistribution/review/pages/page-01.png` … `page-14.png`.

What I checked on the PNGs:
- p8: Eqs. 16–20 (siblings, U, new K with its (N+ε)U identity, new V_i with the NN_50 subscript, deletion pool). Labels and the "Section 3.2" reference resolve.
- p9: the baseline-set text, Eq. 21 (Δ^del), Cohorts, the Preliminary audit paragraph with Eq. 22 (Λ̄^0, r^0, Mem^0 on one line with number), the removal-set bullets (R^vis points to ΔV matching, R^hv cites Eq. 22), and Eq. 23 (ΔU) with the inline ΔK formula.
- p10: the ΔV-matching paragraph, Eqs. 24–28. X^base fits on one line. X^sem is two lines with a vertically centred number (26). The linking pair now uses ΔK.
- p13: Table 3 has the new baseline row, and its totals read 949 and 326. Appendix A text.
- I found no "??" on any page.

## 3. Bibliography check

The `.bib` is unchanged from iteration 1 (17 entries). I took author lists and titles from the arXiv API (`export.arxiv.org/api/query`). DBLP returned HTTP 429 and then a bot wall, so I confirmed venues on the publisher or proceedings pages instead.

| Key | Entry | Checked against |
|---|---|---|
| feldman2020memorization | Feldman, STOC 2020 | arXiv:1906.05271 (https://arxiv.org/abs/1906.05271); ACM DL https://dl.acm.org/doi/10.1145/3357713.3384290 |
| feldman2020neural | Feldman & Zhang, NeurIPS 33, 2020 | arXiv:2008.03703; https://proceedings.neurips.cc/paper/2020/hash/1e14bfe2714193e7af5abc64ecbd6b46-Abstract.html |
| baldock2021difficulty | Baldock, Maennel, Neyshabur, NeurIPS 34, 2021 | arXiv:2106.09647; https://proceedings.neurips.cc/paper_files/paper/2021/hash/5a4b25aaed25c2ee1b74de72dc03c14e-Abstract.html |
| carlini2022lira | Carlini, Chien, Nasr, Song, Terzis, Tramèr, IEEE S&P 2022, pp. 1897–1914 | arXiv:2112.03570; NDL record https://ndlsearch.ndl.go.jp/books/R100000136-I1360020313803156736 |
| carlini2022onion | Carlini, Jagielski, Zhang, Papernot, Terzis, Tramèr, NeurIPS 35, 2022 | arXiv:2206.10469; https://proceedings.neurips.cc/paper_files/paper/2022/hash/564b5f8289ba846ebc498417e834c253-Abstract-Conference.html |
| borkar2025ripple | Borkar, Jagielski, Lee, Mireshghallah, Smith, Choquette-Choo, Findings of ACL 2025 | arXiv:2502.15680; ACL Anthology 2025.findings-acl.959 |
| cohen2026undeleted | Cohen, Kohen, Nissim, Stemmer, FORC 2026, LIPIcs 368, 17:1–17:18 | arXiv:2602.16697; https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.FORC.2026.17 |
| ebrahimpour2026class | Ebrahimpour-Boroojeny, Wang, Sundaram, arXiv:2506.20893 v5 (2026) | arXiv:2506.20893. Cited as a preprint because I could not confirm a peer-reviewed venue. |
| lee2015dsn | Lee et al., AISTATS 2015, PMLR 38 | arXiv:1409.5185; https://proceedings.mlr.press/v38/lee15a.html |
| yan2015hdcnn | Yan et al., ICCV 2015 | arXiv:1410.0736; https://openaccess.thecvf.com/content_iccv_2015/html/Yan_HD-CNN_Hierarchical_Deep_ICCV_2015_paper.html |
| zhu2017bcnn | Zhu & Bain, arXiv 2017 | arXiv:1709.09890 |
| bertinetto2020mistakes | Bertinetto et al., CVPR 2020 | arXiv:1912.09393; CVF open access |
| chen2020simclr | Chen, Kornblith, Norouzi, Hinton, ICML 2020, PMLR 119 | arXiv:2002.05709; https://proceedings.mlr.press/v119/chen20j.html |
| northcutt2021labelerrors | Northcutt, Athalye, Mueller, NeurIPS 2021 D&B | arXiv:2103.14749 (journal_ref) |
| krizhevsky2009cifar | Krizhevsky, U. Toronto tech report 2009 | https://www.cs.utoronto.ca/~kriz/learning-features-2009-TR.pdf |
| vanhorn2018inat | Van Horn et al., CVPR 2018 | arXiv:1707.06642; CVF open access |
| he2016resnet | He, Zhang, Ren, Sun, CVPR 2016 | arXiv:1512.03385; CVF open access |

**Unresolved citations: none.** The only caveat is that Ebrahimpour-Boroojeny et al. 2026 is cited as an arXiv preprint. The companion study is described anonymously and has no bib entry.

## 4. Coverage notes for the reviewer

- Equation numbering shifted. The new equations are K (18), V (19) and the preliminary audit (22). Δ^del is now Eq. 21, ΔU Eq. 23, and the prediction equations are Eqs. 24–28. All references use `\ref`, so the cross-references updated automatically.
- Mapping to the source draft: draft Eq. 22 (support S) → paper Eq. 17 (U). Draft Eq. 25 (ΔS) → Eq. 23. Draft Eqs. 26–30 → Eqs. 24–28. K, V and the preliminary-audit quantities are additions. The draft's §2.10.2 ΔV-matching clause is now implemented in the "ΔV matching" paragraph.
