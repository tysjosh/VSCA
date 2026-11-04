# Symbolic learning for LLM accuracy: what works, why VSCA v1 didn't, and a VSCA v2 that could

*Written 2026-11-04. Literature figures are as reported by their authors (I haven't re-run them), and the sources are paraphrased for licensing compliance. Our own numbers come from the VSCA runs on Qwen2.5-1.5B.*

---

## 1. First principles: when does a symbolic signal raise accuracy?

A symbolic component (a verifier, executor, prover or program) improves a model only through what it changes in training or decoding. Across the literature, one rule explains most of the gains and failures:

> **Gain ≈ (how often the signal fires) × (how much it changes the update or decision there) × (how much those places matter for errors the model actually makes).**

That gives four levers:

1. **Placement: on-policy vs. off-policy.** A signal applied to the *reference* solution (teacher forcing) only shapes behaviour at states the model would visit if it were already right. A signal applied to the model's *own* sampled states hits its real mistakes. Most large symbolic-signal gains come from on-policy training: process rewards, RLVR, and step-level preferences built on model samples.
2. **Polarity: positive vs. negative.** Accepting extra correct continuations (positive set expansion) mostly redistributes mass between correct wordings. Penalizing the *specific wrong step* the model takes changes what it does.
3. **Error coverage: what the verifier can see.** An arithmetic checker only sees arithmetic slips. If most errors are wrong-quantity or wrong-plan errors, the verifier is blind to most of the problem.
4. **Density: where the signal lands.** RL gains concentrate on a small set of high-entropy "forking" tokens ([Wang et al., 2025](https://arxiv.org/abs/2506.01939)), so a signal must reach those positions to matter.

## 2. VSCA v1, diagnosed with these levers

| Lever | VSCA v1 | Evidence from our runs |
|---|---|---|
| Placement | Off-policy: certificates on reference prefixes | Loss differs from SFT at < 1 token per answer |
| Polarity | Positive only: adds certified alternatives | Decision-point alternatives are mostly rewordings ("together" for "a") |
| Error coverage | Checks arithmetic and the final answer only | SFT failures: 18% arithmetic slips, **82% correct arithmetic but wrong answer** |
| Density | Claim tokens (no effect) plus 4 decision positions per answer | Claim positions: certified mass 0.007; decision positions: 17% certified |
| **Result** | +1.6 over SFT, **+0.2 over claim-span masking** (3 seeds, test) | The gain comes from span focus, not from verification |

VSCA v1 put a weak, positive-only signal at places the model was already right, using a verifier blind to most of its real errors. The no-go outcome follows directly from that.

## 3. What the literature shows works (2023–2026)

### A. On-policy process supervision with symbolic or verifier credit: the largest reliable gains

- **Step-level self-exploration and first-error preference:** Self-Explore samples from the model, finds the first "pit" step by rollouts, and builds step-level preference pairs. It reports about **+11.6 points on GSM8K** over SFT, averaged across three models ([arXiv 2404.10346](https://arxiv.org/abs/2404.10346)). Step-DPO and Step-Controlled DPO train on a minimally different wrong step ([SCDPO](https://arxiv.org/abs/2407.00782)).
- **Process vs. outcome supervision in small models:** process-only supervision reached **63.7% vs. 53.8%** for outcome-only, with more valid steps ([arXiv 2607.02869](https://arxiv.org/abs/2607.02869)).
- **A symbolic process oracle in RL:** Lean marks locally sound steps and the earliest failing step. First-error credit beats outcome-only rewards on miniF2F and ProofNet ([arXiv 2606.20068](https://arxiv.org/pdf/2606.20068v1)). This is the formal-math version of the idea we need.
- **Critical tokens:** cDPO finds tokens whose substitution flips the outcome, then applies token-level preference at them ([ICML 2025](https://proceedings.mlr.press/v267/lin25j.html)).

### B. Verifying *reasoning*, not just answers, when filtering self-generated data

- VSI/NSRSA checks each arithmetic operation with sympy, plus step-flow consistency and domain constraints, before keeping a sample for self-training. It **rejects about 34% of correct-answer solutions** as lucky guesses on GSM8K, and turning those checks into DPO pairs raises reward accuracy from 46% to 63% ([arXiv 2603.21558](https://arxiv.org/abs/2603.21558)). That's the job our verifier `V_r` already does, applied to model samples instead of references.

### C. Execution grounding: let the symbolic engine do the step it's good at

- PAL ([2211.10435](https://arxiv.org/abs/2211.10435)), ToRA ([2309.17452](https://arxiv.org/abs/2309.17452)) and program-aided distillation ([2305.13888](https://arxiv.org/abs/2305.13888)) all report large gains on arithmetic-heavy math from generating code and executing it.
- NeuroProlog compiles word problems into Prolog and trains jointly on rule translation and program synthesis. It reports **+3.4 to +5.5 points on GSM8K** across 3B–32B models, with execution-guided repair at inference ([arXiv 2603.02504](https://arxiv.org/abs/2603.02504)).
- **The limit:** solvers help mainly when the task has limited implicit reasoning but a large search space ([arXiv 2512.03272](https://arxiv.org/abs/2512.03272)). GSM8K's errors are mostly in *planning*, which an executor alone doesn't fix.

### D. Token-level credit in RLVR

- Gains concentrate at high-entropy forking tokens ([2506.01939](https://arxiv.org/abs/2506.01939)). Recent work refines token credit by reward polarity and entropy ([2604.11056](https://arxiv.org/abs/2604.11056)). **This is where certified sets belong:** at the forks the model actually faces.

### E. Cautionary results

- **Silent negative transfer:** a verifier that helped earlier can keep looking helpful during self-training after its updates stop generalizing ([2606.14629](https://arxiv.org/abs/2606.14629)). Always evaluate held-out.
- Tool-enabled evaluation can *lower* strong thinking models' accuracy even when they make almost no tool calls ([2605.06326](https://arxiv.org/pdf/2605.06326)).

**Synthesis:** the reliable gains come from on-policy samples with first-error localization, negatives at the failing step, and verifiers that check reasoning validity. VSCA v1 had none of these.

## 4. Proposal: VSCA v2, on-policy verified credit assignment

Keep VSCA's core idea, a set-valued target certified by a verifier, with the one-sided error guarantee of Proposition 4. Change *where* it applies, add a *negative* at the error, and make the verifier see *planning* errors.

### 4.1 Pipeline (per training problem, refreshed each epoch)

1. **Sample** k = 8 solutions from the current model with vLLM.
2. **Verify each step symbolically** with a stronger `V⁺`:
   - **Arithmetic validity:** exact rational evaluation (existing `V_c`).
   - **Grounding / dataflow:** every operand must be a quantity from the problem or the result of an earlier step. This catches made-up numbers and reused wrong intermediates, the "wrong quantity" class.
   - **Final answer** matches gold (existing `V_r`).
   - **Lucky-guess filter:** a correct answer with any failed step counts as *not* verified (VSI-style).
3. **Find the first error.** For a failed sample, take the earliest step that fails `V⁺`. If every step passes but the answer is wrong (a planning error), estimate step values by Monte Carlo: from each step boundary, run M = 4 rollouts and verify them. The first error is the step where the estimated value drops the most, as in Self-Explore and Math-Shepherd.
4. **Certify the fork.** At the prefix just before the first error, propose the model's top-k next tokens and certify each by verified rollout. This is VSCA's `A_t`, applied **at the model's own failure state**.
5. **Loss at the fork:**

   L_fork = −log Σ_{a∈A_t} π(a | h) + β · log(1 − π(e | h))

   where `e` is the erroneous token the model chose. The first term is VSCA's set-valued NLL; the second is unlikelihood on the verified-wrong choice.
6. **Elsewhere:** standard SFT on verified-correct samples, which is the RFT part, plus optional SFT on references.

### 4.2 Why this should fix each failed lever

- **Placement:** the signal lands on states the model *actually reaches*, at its *first mistake*.
- **Polarity:** an explicit negative on the token that started the error, plus a positive set of certified repairs.
- **Coverage:** grounding and MC values reach planning errors, the 82%, not only arithmetic slips.
- **Density:** one or two high-value forks per failed sample, chosen by verification rather than by an entropy budget.
- Proposition 4 still holds for the positive set. False certificates remain the main risk, so keep the certificate audit.

### 4.3 Novelty, positioned honestly

Close prior work: Self-Explore and Step-DPO (step-level preference at the first pit), cDPO (critical tokens), VSI/NSRSA (symbolic filtering), Lean process RL (first-error credit). The claim v2 can defend: **a single proper objective that combines a symbolically certified positive set with a verified negative at the model's first error, plus a grounding verifier that catches planning errors without a learned reward model.** Its value has to be shown against those baselines, not assumed.

## 5. Experiment plan (pre-register before running)

### 5.1 Arms (Qwen2.5-1.5B, LoRA, same budget)

| Arm | What it isolates |
|---|---|
| SFT, claim-span masking | Existing baselines (already have 3 seeds) |
| **RFT-verified:** SFT on samples passing `V⁺` | Value of symbolic filtering alone (VSI-style) |
| **Step-DPO-style:** preference pairs at the first error, no certified set | Value of first-error negatives alone |
| **VSCA v2 (full)** | Certified positive set plus negative at the fork |
| VSCA v2 without grounding (`V⁺` = arithmetic + answer) | Value of the planning-error verifier |
| VSCA v2 with uncertified sets | Value of verification itself |

### 5.2 Go/no-go pilot (3 seeds; ~1.5 reservations)

- **Cost estimate (L40S):**
  - Sampling 8 × 6,726 problems with vLLM: ~1 GPU-hour per refresh.
  - MC rollouts at the forks of failed samples, about 40% of 54k: ~2–3 GPU-hours.
  - Verification: CPU.
  - Each training run: ~25 minutes.
  - Total ≈ 12 GPU-hours for 4 new arms × 3 seeds with one refresh.
- **Rule:** VSCA v2 must beat **both** RFT-verified and claim-span masking by ≥ 1.0 point (GSM8K test, greedy, paired bootstrap). If RFT-verified alone matches it, the contribution is the filter, not the certified sets.
- **Secondary:** SVAMP transfer, pass@8, and valid-trace rate. These test whether the reasoning improved, not only the answers.

### 5.3 Two cheap diagnostics to run first (≤ 1 GPU-hour total)

1. **Grounding-verifier coverage:** on the 200 labelled SFT failures (once you've labelled `error_taxonomy_sft.csv`), measure the share of "wrong quantity" errors that `V⁺` flags. **If it's under ~40%, build a better verifier before training anything.**
2. **Fork recoverability:** at first-error prefixes, how often does at least one top-k alternative have a verified continuation? If it's rarely, the certified set will be empty at the forks, and v2 reduces to Step-DPO.

## 6. Risks and how to defuse them

| Risk | Defence |
|---|---|
| GSM8K ceiling and Qwen math contamination | Add Llama-3.2-1B and MATH-500 subsets; report SVAMP |
| The grounding check is too strict (rejects valid reformulations) | Audit 200 rejections by hand; allow derived constants (e.g. 60 min/h) from a whitelist |
| The MC first-error estimate is noisy | M = 4 rollouts, a value-drop threshold, and skip forks with uncertain values |
| Gains come from extra sampled data, not credit assignment | RFT-verified uses the same samples; this is the key control |
| Silent negative transfer across refreshes | Evaluate held-out after every refresh; stop when held-out flattens |

## 7. Bottom line

The way to get a real gain from symbolic learning is to **verify the model's own reasoning, locate its first verified mistake, and put both a negative and a certified correction exactly there.** Accepting extra correct wordings on reference text, which VSCA v1 did, can't move accuracy much. The most likely single source of gain is the **grounding verifier for planning errors**, because that's where 82% of the failures are. It's also the cheapest thing to test first.
