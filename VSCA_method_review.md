# Review of VSCA: "Verifiable Semantic Credit Assignment for Language Model Post-Training"

Source reviewed: Overleaf project `6ae4fd18e6b1331c65d752c7`, `example_paper.tex` and `example_paper.bib` (downloaded 2026-10-30).

## 0. The method in one paragraph

VSCA extracts atomic claims from a verified SFT target and aligns them to token spans. For each claim it builds one hand-made counterfactual (for example `84 -> 85`) and sets a binary sensitivity score `s = 1` when the original verifies and the counterfactual fails. It mixes `s` with a Rho-1-style excess-loss difficulty `d`, using `q = αd + (1-α)s + γds`. Top-K tokens under complete-span constraints get the SFT loss. A reference-free, length-normalized pairwise loss (`L_cf`) is optionally added to prefer the verified span over its counterfactual.

## 1. Overall assessment

The paper asks a good question: does verifier-grounded importance carry information that loss- and attention-based token selection lack? The protocol is careful about compute accounting, matched budgets and shuffled controls, more so than most papers in this area.

The method itself has a structural problem. From first principles, the semantic signal it defines is close to degenerate. The counterfactuals come from the wrong distribution. And the signal points at the tokens where small models make the fewest consequential errors. My prediction is that, as written, VSCA-Selection will match the "unscored claim-span" control within noise. The planned ablations would then end up showing that verification adds nothing. That would be an honest result, but a weak paper.

Section 6 gives a reformulation that keeps the paper's intuition, comes from an explicit objective, and removes most hyperparameters. I recommend adopting it before running the full protocol.

Readiness: as a proposal, the motivation and evaluation design are close to workshop quality. The method needs a redesign before it is a main-track ICML submission.

---

## 2. Core technical problems (ranked by severity)

### C1. The binary sensitivity `s^claim` is near-degenerate under every verifier regime the paper allows

Equation (5) is `s = 1[V(y)=1 ∧ V(ỹ)=0]`. The candidate verifier designs are:

1. **Local claim verifier** (does `12×7=84` hold?). The counterfactual is defined as a change to "one semantically relevant component" of a *true* atomic claim. For arithmetic or a functional relation, any such change makes the claim false by construction. So `s = 1` for essentially every accepted claim. The score becomes `1[claim extracted and counterfactual accepted]`. That is the unscored claim-span control plus an acceptance filter.
2. **Final-answer verifier with no propagation.** You edit an intermediate value but leave the rest of the response alone. The final answer string is unchanged, so `V(ỹ) = 1` and `s = 0` for every intermediate claim. Only the final-answer span gets `s = 1`, which is the AnswerOnly baseline.
3. **Final-answer verifier with downstream recomputation.** The edit is no longer minimal, and arithmetic chains are almost always sensitive to every value. So `s ≈ 1` again for all claims on the solution path.

In all three regimes, `s` is (almost) a deterministic function of claim extraction. The counterfactual step adds variance only through the acceptance filter, and the acceptance filter correlates with extraction quality, not with semantic importance. The one interesting case is a redundant or irrelevant step whose change does not affect correctness. Only regime 3 detects it, and redundant steps are rare in GSM8K reference solutions.

**Required:** report the empirical distribution of `s` over accepted claims per task. If more than ~95% are 1, the "counterfactual semantic importance" framing does not hold, and the method should be described as claim-span selection.

### C2. The counterfactuals are drawn from the wrong distribution

Credit assignment is about which *model decisions* change the outcome. A hand-made perturbation (`84 -> 85`, swapping an entity) asks whether the verifier is sensitive to this text edit. That is a property of the verifier and the data, not of the model. It is fixed before training and the same for every model.

The relevant question is whether, at position t, the model puts probability mass on alternatives that break correctness. That depends on `p_θ(·|h_t)`, and it changes during training. A token where the verifier is sensitive but the model already puts 0.999 on the gold token needs no extra credit. Section 6 shows that this is exactly what the gradient of an expected-correctness objective says.

The paper's own interaction term `γ d s` hints at this: it is a rough proxy for "the model is likely to err here, and the error matters". But it multiplies a binary verifier score by a loss gap measured against a reference model. That is a poor estimate of the actual quantity, the probability mass on harmful alternatives.

### C3. The signal likely targets where errors are not

In GSM8K-style chains, the result token of `12×7=` is the most predictable token in its neighborhood once the expression is written. The decisions that determine correctness are earlier: which quantities to combine, which operation to apply, what to compute next. Those are the **planning tokens** in the prose and in the left-hand side of equations.

A local substitution with a fixed continuation cannot detect that a planning token matters, because its effect runs through the continuation. Only rollouts can measure that (Monte Carlo value estimation, as in Math-Shepherd or VinePPO). The method's semantic score is therefore systematically blind to the tokens that most plausibly cause failures.

**Required before anything else:** an error taxonomy for the base model on about 200 GSM8K validation failures, split into arithmetic slip, wrong operation or quantity, misread problem, and format. If arithmetic slips are a minor share, which I expect for Qwen2.5-1.5B, the central hypothesis has weak support before training starts.

### C4. The difficulty signal `d` is undefined, and with the obvious choice it collapses to zero

Equation (6) needs a reference model, and the paper only says the reference "uses only permitted training data". In Rho-1, the reference is trained on high-quality data, so `ℓ_θ − ℓ_ref > 0` marks learnable tokens the student has not learned yet. Here the student starts from the base checkpoint. The two obvious choices are:

- **Reference = base model.** Then `e = 0` at step 0. As the student fits the training targets, `ℓ_θ < ℓ_ref` on most tokens, so `max(0, e) = 0`. `d` is zero or noise almost everywhere, and VSCA reduces to `s` alone.
- **Reference = model fine-tuned on the same verified targets.** This works like Rho-1, but it costs a full SFT run, which must be counted. It also risks circularity: the reference has memorized the training tokens, so `e` mostly measures how far the student is behind the reference, not whether the tokens are useful.

`Normalize(·)` is also unspecified: per-sequence, per-batch, min-max or rank. This matters because `d` is mixed with a binary `s` (see C6).

### C5. The contrastive loss is small, saturates fast, and overlaps with CE

Take the paper's example. `84` and `85` share the prefix `h`, and Qwen2.5 tokenizes digits individually, so the spans differ only in the last token. Then:

`S(z|h) − S(z̃|h) = (1/|z|)·(log p("4"|h,"8") − log p("5"|h,"8")) = (logit_4 − logit_5)/|z|`

This leads to four problems:

- **Saturation.** If the model already has `p("4") = 0.95` and `p("5") = 0.001`, then with `|z| = 2` and `τ = 1` the margin is `ln(950)/2 ≈ 3.4`. The gradient scale is `σ(−3.4) ≈ 0.03`. For off-by-one numeric counterfactuals the loss is mostly inactive. It only acts where the model is confused between exactly that pair, and hand-made counterfactuals rarely pick that pair.
- **Overlap with CE.** The CE gradient on the gold token already pushes down every alternative in proportion to its probability, including `"5"`. `L_cf` adds extra push on a single hand-picked alternative. That is unlikelihood training (Welleck et al., 2019) restricted to one negative.
- **Length normalization is an artifact here.** Shared tokens inside the span (the `"8"`) add zero to the difference but enlarge `|z|`. The margin then depends on how many shared tokens the span includes, which is a tokenization detail. For different-length alternatives (`84` vs `840`, `Paris` vs `Saint-Étienne`), per-token average log-probabilities are not comparable probabilities. The objective then has a free length bias, as discussed in the SimPO and length-normalized DPO literature.
- **No reference term.** This is SimPO-style, reference-free preference learning on local pairs. Such objectives can lower the probability of the chosen response along with the rejected one (likelihood displacement). The paper adds `L_select` to anchor it, but needs to measure gold-token log-probability on held-out data to show the anchor works.

What `z` covers is also unclear: the whole claim `12×7=84` or only the changed token. And "applied only when … a verified interpretation" exists is not operational. The "safe label-consistent random-contrast control" is not defined.

### C6. The score combination reduces to a threshold, and γ is nearly redundant

With binary `s`:

- claim token (s=1): `q = (α+γ)·d + (1−α)`
- non-claim token (s=0): `q = α·d`

Global top-K therefore ranks claim tokens above non-claim tokens whenever `(1−α) > α·max d − (α+γ)·d`, then ranks by `d` within each tier. The three-parameter family `(α, γ, K)` is effectively a priority rule plus one crossover point. γ only matters near the tier boundary.

This makes the planned α/γ grid search and the NoInteraction ablation close to uninformative. The paper should say plainly that, with binary `s`, the policy is "claim spans first, then excess loss", and simplify accordingly.

Two related issues:

- **Coverage bias from the fallback.** Examples with no verified claims get scores `αd < 1−α`. Under a global budget they are systematically under-selected. Extraction fails more often on problems with fractions, percentages, units or unusual formatting, so supervision is withdrawn exactly where the data is harder. The per-example coverage floor reduces this but turns it into another hyperparameter. Report selection rate as a function of problem difficulty.
- **The span-constrained top-K is a knapsack.** State the algorithm (greedy by mean span score?) and the scope of the budget (per sequence, per batch or global).

### C7. The "budget" is not binding, so the motivation needs restating

The paper correctly says that masking saves no FLOPs. But then no supervision budget is scarce in SFT, and "allocating a limited post-training budget" has no physical meaning. Masking can only help through one of three mechanisms:

1. removing gradient noise from high-entropy tokens with many valid choices (paraphrase positions);
2. reducing overfitting or forgetting by leaving the base model's distribution untouched on unsupervised positions;
3. reweighting the gradient toward important positions.

Each mechanism suggests a cleaner method and a specific test. (1) points to set-valued targets (Section 6). (2) predicts gains in held-out general capability: measure MMLU or IFEval drift against full SFT. (3) points to soft upweighting, `w_t = 1 + β s_t`, rather than hard masks. **Soft upweighting is the most natural competitor and it is missing from the baselines.**

**Confound in Eq. (9).** Normalizing by `Σ m` means that at 25% selection, each selected token gets 4× the per-token weight it has under full SFT at the same learning rate. AdamW partly absorbs this, but the mix of gradient directions still changes. The ratio sweep therefore confounds selection with effective learning rate. Either tune the learning rate per ratio and method under the same tuning allowance, or normalize by total tokens and report both.

---

## 3. Specification and notation issues

- Eq. (3)/(5): `V(x_i, y_i, E_i) = 1` is true by assumption, since `y_i` is a verified target, so the conjunction is vacuous. `V` is defined on whole responses but used to judge claims. Define `V_resp` and `V_claim` separately and say which one Eq. (5) uses on each task.
- `a_{i,j,t} ∈ [0,1]` is a soft alignment, but the text requires complete spans. Say when an alignment is fractional, or make it binary.
- GSM8K reference solutions include calculator annotations (`<<48/2=24>>`). Say whether they are kept in targets. If they are, claim extraction and verification are trivial and the claim "deterministic verification is practical" is carrying little weight. If they are stripped, the extractor's precision and recall need reporting.
- "Contextual validity" vs "arithmetic validity" is the important distinction for GSM8K (a true equation that computes the wrong quantity), but no procedure for checking contextual validity is given. Without one, `s` measures arithmetic validity only. That ties back to C3.
- The ProofWriter counterfactual for a derived fact is usually negation. Under the open-world assumption, the negation of a provable fact is "false" but the negation of an unknown is "unknown". Specify the rule.
- Line 63 of the source has `	extbf{` with a literal tab. A `\t` got interpreted when the file was written, so the bold command is broken.

## 4. Experimental design

**Backbone.** Qwen2.5 base models already score high on GSM8K out of the box, and their math pretraining data is heavy and partly synthetic. SFT gains on GSM8K-train are small and noisy at this starting point. Qwen2.5 models are also known to respond to weak or even spurious training signals in math RL settings. A method whose effect is a small reallocation of SFT loss is exactly the kind of result that may not transfer. Add at least one non-Qwen backbone (Llama-3.2-1B/3B or OLMo-2-1B), ideally one with a weaker GSM8K prior.

**Ceiling on ProofWriter.** Fine-tuned models reach very high in-distribution accuracy on ProofWriter. Use held-out depth (train ≤ 3, test 5) under the open-world assumption as the primary split, or add a harder logic set.

**Statistical power.** GSM8K test has 1,319 items. At about 70% accuracy the binomial SE per run is about 1.3 pp. For paired comparisons with about 10% discordant items, the SE of the difference is about 0.9 pp, before LoRA seed variance (often 0.5 to 1.5 pp on GSM8K). Token-selection gains in the literature are often 1 to 3 pp. Three seeds is underpowered for the confirmatory comparison. Pre-register the minimum effect size of interest, compute the seed count from a pilot variance estimate, and use paired item-level tests (McNemar or paired bootstrap) pooled across seeds.

**Missing baselines, in priority order:**

1. Unscored claim-span selection at the same budget. This is listed as an ablation, but given C1 it is the decisive comparison and should be the primary confirmatory one.
2. Soft upweighting of claim tokens, `w = 1 + β·1[claim]` (see C7).
3. TrimSFT (logit-gap weighting). It is cited, targets math SFT directly, needs no reference model, and is missing from the baseline list.
4. Full SFT plus unlikelihood on the same counterfactual tokens. This isolates whether the sigmoid pairwise form matters.
5. cDPO / critical-token methods (Lin et al., ICML 2025). These identify outcome-critical tokens in GSM8K/MATH reasoning with contrastive estimation, which is the closest prior work to the paper's central idea.

**Falsifiable predictions.** State them up front:

- VSCA-Selection > claim-span ≈ Shuffled means the counterfactual verifier carries information.
- VSCA-Selection ≈ claim-span means the gain, if any, comes from span priors, not verification. This is my expectation given C1.
- VSCA-Selection ≈ Rho-style means the semantic term adds nothing beyond difficulty.

## 5. Related work and bibliography

**Missing, and close to the core idea:**

- *Critical Tokens Matter / Token-Level Contrastive Estimation* (cDPO), Lin et al., ICML 2025 (arXiv 2411.19943). Outcome-critical tokens in math reasoning found by forced decoding and contrastive models. The novelty claim must be positioned against this.
- Process supervision and value estimation: *Let's Verify Step by Step* (Lightman et al., 2023), Math-Shepherd (Wang et al., 2023), VinePPO (Kazemnejad et al., 2024), PRIME implicit PRMs (Cui et al., 2025). These are the rollout-based answer to "which steps change correctness".
- Step-DPO (Lai et al., 2024): step-level preference pairs with a minimally different wrong step, very close to `L_cf`.
- Unlikelihood training (Welleck et al., 2019) and SimPO (Meng et al., 2024): the direct ancestors of the form of `L_cf`.
- Work on high-entropy "forking" tokens in reasoning RL (Wang et al., 2025, "Beyond the 80/20 rule"): the planning-token argument in C3.

**Bibliography errors I checked against the live pages:**

| Key | Problem | Correct |
|---|---|---|
| `rho1` | First author listed as "Zhiqiu Lin" | Zhenghao Lin et al. |
| `ssToken` | First author listed as "Yuxuan Qin"; year 2026 | Xiaohan Qin et al.; arXiv v1 is Oct 2025 |
| `trimSFT` | Wrong title and "Anonymous" authors | "Which Tokens Should SFT Actually Learn? A Token-Trimming Perspective on Mathematical Reasoning", Jia, Zhang, Xu, Diao, Wang, Vosoughi, Findings of EMNLP 2026 |
| `beyondTokens` | Wrong title ("Concept-Level Prediction for Language Model Training") and "Anonymous" authors | "Beyond Tokens: Concept-Level Training Objectives for LLMs", Iyer, Somani, Guo, Jurafsky, Shani, EACL 2026 (short) |
| `verifiableProcess` | No authors, venue or link; I could not confirm it exists | Find the actual paper or remove the citation |
| `gsm8k` | `@inproceedings` with booktitle "arXiv preprint" | Use `@article` / `@misc` |

Note that Iyer et al. (EACL 2026) make the paper's exact motivating argument: token-level NLL penalizes semantically equivalent alternatives. Engage with it directly rather than citing it in passing.

---

## 6. A first-principles reformulation (recommended)

The objective VSCA is implicitly trying to optimize can be written down directly.

**Setup.** Teacher forcing on the gold prefix `h_t`. Let `π = p_θ(·|h_t)`, `g` the gold token, and `Q_t(a) ∈ [0,1]` the verified correctness after substituting token `a` at position t (for a local verifier, the correctness of the claim; `Q_t(g) = 1`). The one-step-deviation expected correctness is

```
J_t(θ) = Σ_a π(a) · Q_t(a)
```

**Gradient in logits.** `∂J_t/∂z_b = π(b) · (Q_t(b) − J_t)`.

Three things follow:

1. **Harmless alternatives are not penalized.** Tokens with `Q_t(b) = 1`, such as paraphrases or equivalent forms, are pushed up together with the gold token. CE pushes them down. This is exactly the paper's "Paris vs. a rephrasing" intuition, and it falls out of the objective without any token selection.
2. **The per-position credit is `1 − J_t`**, the probability mass the model puts on harmful alternatives. When the model is already correct or all alternatives are harmless, the gradient is zero, so no external budget or mask is needed. This is the principled version of the `γ·d·s` term: it is model-dependent, continuous and self-annealing.
3. **The negatives are the model's own.** The gradient pushes down `b` in proportion to `π(b)·(1 − Q_t(b))`, which is the unlikelihood that `L_cf` approximates with a single hand-made negative.

**Practical form: set-valued cross-entropy.**

```
L_t = − log Σ_{a ∈ A_t} π(a),     A_t = { a : Q_t(a) = 1 }
```

At positions with no applicable verifier, use `A_t = {g}`, which is ordinary CE, or a fixed weight. At claim positions, evaluate `Q_t` only on the top-k tokens under `π` (k around 5 to 20, cheap for arithmetic and structured facts). Then:

- it removes α, γ, K, λ_cf, τ, the counterfactual generator and the reference model;
- full SFT is recovered exactly when `A_t = {g}` everywhere, which makes the ablation clean;
- the remaining research question is precise: does replacing `{g}` by the verified-equivalence set `A_t`, and weighting by harmful mass, improve correctness over CE, TrimSFT and claim-span upweighting?

**Planning tokens (C3).** Local `Q_t` cannot see them. The faithful extension estimates `Q_t(a)` by rolling out from `h_t ⊕ a` and verifying the final answer (VinePPO-style MC). It is expensive, so restrict it to the high-entropy positions where `π` actually spreads its mass, and compare against cDPO, which addresses the same question.

**Honest scope.** For GSM8K arithmetic results, `A_t` is usually `{g}`, because only one digit is correct, so the set-valued CE equals CE there and the method's effect concentrates on the high-entropy positions. That is also where the paper's motivation says the effect should be. If no gain appears there, the hypothesis has been tested cleanly and rejected.

## 7. Suggested order of work (cheap kill-tests first)

1. **Error taxonomy** of the base model on about 200 GSM8K validation failures (C3). About one day of work, and it can falsify the premise.
2. **Distribution of `s`** over accepted claims on GSM8K and ProofWriter (C1). Hours, CPU only.
3. **Gradient activity of `L_cf`:** the fraction of training pairs with `σ(−margin/τ) > 0.1` at initialization (C5). One forward pass.
4. **Pick the reference model** and plot the distribution of `d` over the first epoch (C4).
5. **Pilot with one seed:** full SFT vs claim-span vs VSCA-Selection vs set-valued CE, to estimate effect sizes and variance for the power calculation.
6. Only then run the full RQ1 to RQ5 protocol.
