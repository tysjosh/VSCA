# Review: iteration 2 (DONE MEANS item 4)

Inputs read: `paper/iclr2025_conference.tex` (full), `source_draft.md`, companion `semantic_grouping/conference_101719.tex` (Method, Experimental Design, Inference; read-only), `review/verification.md` (iteration 2), the iteration-1 `review.md`/`review.json`, and rendered pages 9 and 10. I did not re-run the compile. The .bib did not change in this iteration, and I relied on the iteration-1 check of it.

## Prior findings: verification of the fixes
1. **Δ^del attack configuration: resolved.** In §3.10 each Study B condition now has its own baseline set of M_B models (l.~318–320). That set is scored with the same leave-one-model-out procedure, variance rule and α as the deletion sets. The Study A M=32 set is explicitly excluded. Eq. 21 names the source of both terms. The pool-size effect that remains is common to every set of size |R|, and the paper reads positive values relative to R^rand, which P4 tests. The §3.5 variance rule now uses per-target counts (|IN_i∖{m}|, |OUT_i∖{m}| ≥ 15). That gives per-record variances at M=32 (15/16) and pooled variances at 16 (7/8). Both outcomes are well defined. Table 3 adds a baseline row of 32 runs and 11 GPU-h. I recomputed the table under its timing assumption: 384+16+5+32+384+128 = 949 runs, and 134.4+5.6+2.8+11.2+129.0+43.0 = 326 GPU-h. The §4.2 text matches.
2. **Degenerate ΔU: resolved.** K_i^G (Eq. 18) and V_i (Eq. 19) are defined. ΔK^{G_true} is the primary support change. It is negative under R^sem and exactly 0 for T_q under R^vis, because R^vis draws only from classes outside C_q ∪ Y_q^sib. It is small under R^rand. The linking pair and X^sem (Eq. 26) use ΔK, and U/ΔU are kept as the source defines them, with the zero-expectation caveat stated. The ΔV-matching paragraph implements the source §2.10.2 clause: a same-class swap search that leaves per-class counts unchanged, uses only features, has a stated tolerance and cap, and reports any remaining gap. The draft's similar-ΔV / different-ΔK contrast now feeds P3.
3. **R^hv and r_i^0 definitions: resolved.** Eq. 22 defines the continuous score Λ̄_i^0 with a tie-break by record index, r_i^0 at α=0.1%, and Mem_i^0, all on an independent M^0 set. At M_0=16 there are 40,000×8 = 320,000 nonmember pairs, which gives about 320 expected false positives and satisfies the ≥100 rule. Study B at 0.1% has 38,400×8 = 307,200 pairs, so it also passes.
4. **RQ4 influence-related measures: resolved.** The RQ4 wording is restored, and Mem_i^0 and Λ̄_i^0 enter X^base (Eq. 25) as controls taken from the preliminary audit. These are disjoint from the sets that produce the target.

## Re-check of affected paths and the remaining requirements
- **Membership design:** each record is IN for exactly M/2 models. Leave-one-model-out scoring, the pooled threshold, and P1 equal to the mean of Δ^sup all still hold. Every deletion, baseline and preliminary set has its own membership matrix and calibration.
- **Deletion conditions:** all sets have the same size and are chosen before any target model is trained. They exclude T_q, and their class composition is specified. Q, η, M_B and M_0 are parameters, and Table 3 instantiates them (η=0.04, |R|=1,600, 100 records per class).
- **Companion consistency:** I checked the paper's W(G) (Eq. 7), objective and λ_s/ρ, C_G·S_G, PD with k=20 and the fixed 10%-budget probe set, δ, the conditions table and CE-A/CE-B strata against the companion's source. All match. Two differences are deliberate and stated: the shared split changes the probe set from 4,500 to 4,000 images and trains φ̄ on the 40k pool. The 9+9 Experiment B partitions exist at every budget, so the 4+4 privacy subset is valid.
- **Six design choices, added requirement (1)–(5), no-results statement, citation rules:** unchanged since iteration 1 and still satisfied. The source draft's RQ1–RQ4, Studies A/B, threat model, datasets, statistics, threats to validity and Eqs. 1–30 are all accounted for. Coverage of the iteration-2 renumbering is listed in verification.md §4.
- **Compile and render evidence:** verification.md reports exit 0, no undefined references or citations, and no overfull boxes. Pages 9–10 render Eqs. 21–28 cleanly, with no "??".

## Non-blocking notes (no change required for approval)
- In §3.2, l.~143, "They are unchanged; only the split above is shared" can be read as saying that the split is the only shared item, which would contradict Table 2. The intended meaning is probably "only the split differs from the companion's current draft".
- The abstract's summary of RQ4 omits "influence-related measures". The body states RQ4 in full.

## Blockers
None.

Verdict: APPROVED.
