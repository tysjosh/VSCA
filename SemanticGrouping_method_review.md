# Review: "When Does Semantic Grouping Change Fine-Grained Learning Difficulty?"

Source reviewed: Overleaf project `6ad990bb16325edf2ba068d1` ("IEEE Conference Template"), `conference_101719.tex`, downloaded 2026-10-30. The manuscript has a Method, Experimental Design and Discussion section, with no introduction, related work, results or references.

## 0. The study in one paragraph

A CIFAR-100 ResNet-18 gets auxiliary classification heads at stages 2 and 3. Treatments differ only in the auxiliary labels: the curated 20 superclasses (True-Hier), 20 balanced random groups (Random-Hier), 20 balanced clusters of class centroids from a frozen extractor (Visual-Cluster), the 100 fine labels (Flat-Aux), or the superclasses at the final stage only (Final-Hier). Two independent cross-entropy models act as controls: CE-A defines difficulty strata on held-out data, and CE-B is the independent baseline. Outcomes are hard-example accuracy, prediction depth (PD) from k-NN probes on all 8 residual blocks, neighbor purity, and within- vs across-superclass errors. These are measured at four labeled-data budgets. Three hypotheses are tested: H1 (coarse benefit beyond visual grouping), H2 (fine-grained interference) and H3 (larger benefit when labels are scarce).

## 1. Overall assessment

This is a careful design for a small question. The CE-A/CE-B split to avoid regression to the mean is correct and is often missed in the example-difficulty literature. The control set (random, visual, flat, placement) is the right kind of control, and the paper explicitly allows null results.

The main problems are in measurement and inference, not in the controls:

1. Several outcomes are mechanically produced by the treatment, so H1 and part of H2 can be confirmed by construction.
2. The treatment of interest, the human hierarchy, is a single grouping (n = 1). It is compared against three draws of each control, so "semantic content" cannot be separated from "one particular partition".
3. Visual-Cluster is not a neutral matched control. It is the grouping with the most visually confusable siblings, so the True-Hier vs Visual-Cluster contrast mixes "semantic" with "less within-group similarity".
4. The coarse labels are a deterministic function of the fine labels. Any effect therefore comes from the partition acting as a prior or an optimization bias, and H3's rationale has to be stated in those terms.

Each problem has a fix that keeps the study's structure and makes the conclusions much harder to dispute.

Readiness: the design is close to a solid workshop paper once the fixes in Section 6 are adopted. As a conference paper it also needs an introduction, related work, references, a fully specified training recipe, and either a dataset with a real taxonomy or a narrower title.

---

## 2. Core problems (ranked by severity)

### C1. Several outcomes are optimized directly by the treatment

**H1 is close to a tautology as measured.** True-Hier trains linear heads at stages 2 and 3 to predict the 20 true superclasses. Visual-Cluster trains the same heads on a different partition. Coarse-category probe accuracy at those layers, measured with the *true* superclasses, will favor True-Hier by construction, because it is the only condition trained on that target. Finding that True-Hier wins on "early coarse-class separability beyond Visual-Cluster" would say nothing about semantics.

Fix: measure each method's coarse separability on its *own* grouping and compare the gains over CE-B. Or use a grouping-agnostic measure such as the class-similarity structure (RSA against a fixed reference), or fine-class separability.

**PD and blockwise k-NN accuracy measure where supervision was applied.** Flat-Aux trains stages 2 and 3 for fine-class linear separability, which raises fine-class k-NN accuracy at those blocks and lowers PD almost automatically. The coarse heads do the same to a lesser degree. "Lower prediction depth" under an intermediate-head intervention is therefore not evidence that examples became easier. It shows that the layers under the heads were trained to separate classes.

Fix: report PD for the *final* representation pathway only, computed after removing the aux heads. Treat stage-2/3 probe results as manipulation checks, not as outcomes. Also add a frozen-backbone linear probe of the fine classes at each block, so that "information present" can be told apart from "information linearly exposed".

**k-NN probes depend on the reference set size.** The probes and neighbor purity use the training images of each budget as the neighbor set. At 10% budget there are 45 images per class, versus 450 at 100%, so k-NN accuracy and purity shift with budget for reasons unrelated to the representation. This directly confounds H3, which is a method × budget interaction. Use a fixed-size reference set, the same images at every budget, sampled from the smallest budget's training split. Or report linear probes trained on a fixed probe set.

### C2. The true hierarchy is a single draw, so "semantic content" is not identifiable with 3 control draws

The unit of inference for H1 is the *grouping*, not the training seed. True-Hier is one partition, and Visual-Cluster and Random-Hier each get three. Suppose True-Hier beats the mean of three visual groupings. That is compatible with "semantic structure helps" and also with "this happens to be one of the better partitions". Five seeds reduce training noise but say nothing about partition-to-partition variance.

Fix: treat True-Hier as one point in a distribution of matched partitions and report its **rank**. With R = 19 matched partitions per family, the smallest attainable one-sided p is 0.05. Seeds can be reduced to 2 or 3 per partition, because partition variance is the quantity that matters. The paper already says it will add draws if results depend on the draw. This makes that the default and gives an exact test.

### C3. Visual-Cluster is not a neutral control: within-group similarity is confounded with semantics

Visual-Cluster groups classes whose centroids are closest. Its auxiliary heads therefore ask stages 2 and 3 to map the *most confusable* sibling classes to the same output. That is exactly the condition under which H2-style interference is most likely. True-Hier has lower within-group visual similarity: CIFAR-100 superclasses such as "medium mammals" or "large man-made outdoor things" mix visually different classes. True-Hier beating Visual-Cluster could then mean "groups of less similar classes interfere less", with no role for semantics.

The paper acknowledges that visual similarity can carry semantics, but the design still has a single visual point, so the confound cannot be estimated.

Fix: make within-group similarity a measured axis.
- Define `W(G)` as the mean within-group centroid cosine similarity under a fixed extractor.
- Generate balanced partitions spanning `W` from maximally similar (Visual-Cluster) to maximally dissimilar ("anti-visual") by annealing or swap search. Random partitions sit in the middle.
- Fit the outcome as a function of `W` across partitions. The **semantic effect is True-Hier's residual from that curve.**

This turns H1 into one pre-registered quantity with an uncertainty estimate. Also report the agreement between True-Hier and each Visual-Cluster draw (ARI or NMI).

**The extractor is unspecified.** If it is ImageNet-pretrained, Visual-Cluster carries external knowledge, and the contrast becomes "human prior vs ImageNet prior". If it is a CIFAR-100 model trained on all 50k labeled images, then at the 10% and 25% budgets Visual-Cluster uses labels outside the budget. That breaks the paper's "matched labeled-data budget" claim. Choose one option and state it. I'd train the extractor on the budget's own training split, or use a self-supervised extractor on unlabeled CIFAR images, and say which.

### C4. The coarse labels add no information about the images, so H3 needs a different rationale

Each CIFAR-100 fine class belongs to exactly one superclass. The coarse label of every training image is a fixed function `G(y)` of its fine label, so the auxiliary loss adds no new labeled information at any budget. Whatever True-Hier offers comes only from **the partition itself**, a prior chosen by humans, which acts on optimization and inductive bias.

To size that prior: there are `100! / ((5!)^20 · 20!)` balanced partitions of 100 classes into 20 groups of 5. That is about e^225.6, roughly **325 bits** to specify the partition. By comparison, the fine labels at the 10% budget carry about 4,500 × log2(100) ≈ 30,000 bits. Bits are not the right currency for inductive bias, but the comparison shows that H3 cannot rest on "the hierarchy supplies extra supervision when labels are scarce".

The defensible version of H3 is "a fixed prior matters more when the data term is weaker". That predicts an advantage that grows as the budget shrinks *for any good prior*. It is exactly what Visual-Cluster, trained on the budget's own data, does not have. State H3 this way. If the intended mechanism is coarse labels for *extra* images (the usual practical setting), that is a semi-supervised experiment: fine labels on the budget and coarse labels on more images. Either run it or drop the label-scarcity framing.

### C5. The H2 metric is confounded by H1; use a factorization

The paper measures interference as "the fraction of errors that fall in the true superclass". That fraction rises mechanically when coarse accuracy improves, even if sibling discrimination is unchanged.

Write fine accuracy as

```
Acc = C · S,   C = P(ŷ ∈ G(y)),   S = P(ŷ = y | ŷ ∈ G(y))
```

Here `C` is coarse correctness of the fine prediction and `S` is sibling discrimination given the right superclass. The within-superclass error fraction is

```
F = C(1 − S) / (1 − C·S),   ∂F/∂C = (1 − S)/(1 − C·S)² > 0
```

so `F` increases with `C` at fixed `S`. A model that gets more superclasses right will show "more within-superclass confusion" without any interference. This is the same effect that the "better mistakes" line of work reports as a *benefit*.

Fix: make `C` the H1 outcome and `S` the H2 outcome, each with its own contrast. This factorization is the cleanest single improvement available to the paper. Compute it against the true hierarchy for every method. For the control groupings, also compute it against each method's own partition.

The 4/99 reference is correct for uniformly random errors, but no trained model is close to it. Drop it, or keep it in a footnote.

### C6. Model selection is noisy at small budgets, and the training recipe is unspecified

**Validation noise.** At the 10% budget the validation split has 500 images, 5 per class. The binomial SE of validation accuracy around 40% is about 2.2 points. Choosing λ among several configurations with smaller true differences is mostly noise and produces winner's-curse effects that differ across methods. Prefer one of these:
- fix λ per method from a pilot at the 100% budget, with gradient-norm matching (see next point), and use it at every budget;
- report the full λ sensitivity curve per method and budget, with the selected λ as one point.

**Loss scale.** The initial CE is ln 20 ≈ 3.0 for 20-way heads and ln 100 ≈ 4.6 for Flat-Aux. The same λ therefore means different gradient magnitudes. Either match the auxiliary gradient norm at the attached layers or tune λ per method on the same grid, and report the realized gradient-norm ratio.

**Recipe.** Missing details: CIFAR ResNet-18 variant (3×3 stem, no max-pool?), optimizer, schedule, epochs, augmentation, batch size, aux head architecture (GAP + linear, or conv + GAP + linear?) and the exact attachment point. These choices decide the result. Deeper heads absorb the auxiliary task and shield the backbone, and auxiliary-loss gains often shrink under strong augmentation and long schedules. Fix one standard recipe (e.g., SGD with momentum, cosine schedule, 200 epochs, crop + flip) and state it.

"Hyperparameters chosen once with CE": the best weight decay changes strongly between 4,500 and 45,000 images. Select learning rate and weight decay per budget with CE, then share them across methods.

### C7. Scope: CIFAR-100 is not a fine-grained benchmark, and its hierarchy is weak

CIFAR-100 images are 32×32 and its superclasses are a shallow, partly arbitrary grouping. "Fine-grained classification" usually means CUB-200, Stanford Cars, FGVC-Aircraft or iNaturalist, where classes are visually close and the taxonomy is deep and well defined. The title promises more than one CIFAR-100 study can deliver. Either retitle (e.g., "…on CIFAR-100") or add one dataset with a real taxonomy where semantics and visual similarity come apart. An iNaturalist subset with genus or family groupings, or CUB with order or family, would work.

### C8. Hard-example analysis: stratum definition, label noise and cross-fitting

- **Definition of "hard".** Not given: CE-A error, a bottom margin quantile, or PD under CE-A? The primary contrast depends on it. Fix it in advance, and also report a continuous version (outcome vs CE-A margin) so the result does not hinge on a cut-off.
- **Label noise.** CIFAR-100's test set has a few percent of label errors by published estimates, and CE-A's hard stratum will be heavily enriched for them. No intervention can "fix" a mislabeled image, so this compresses effects in the primary outcome. Report a sensitivity analysis that excludes flagged label errors.
- **Cross-fitting.** "Multiple reference/comparator splits are aggregated" should be made concrete. Rotate which CE seed defines the strata, evaluate every other model on those strata, and average. This is cross-fitting, and it makes the regression-to-the-mean correction use every CE run.
- **Primary outcome.** Hard-subset accuracy is a noisier and reference-dependent version of overall accuracy. Keep it, but make overall fine accuracy and the factorized `C` and `S` co-primary. Then a positive result does not depend on the stratification choice alone.

---

## 3. Specification and notation issues

- Eq. (3), PD: the definition requires agreement with the trained classifier, not with the true label (as in Baldock et al., 2021, who should be cited). State k, the distance (cosine?), how block features are pooled (global average pooling discards spatial information more in early blocks), and the reference set. Early-block probes for 100-way CIFAR are near chance, so PD will pile up at high depths. Report its distribution, not only means.
- "Stages 2, 3": say whether heads attach at the output of stage 2 and stage 3. Say whether the stage-3 head for Final-Hier is replaced or removed.
- Final-Hier: a coarse head on the final features is almost redundant with the fine head, since the coarse posterior can be obtained by summing fine probabilities. Say what it is meant to isolate. The useful contrast is probably "true hierarchy at the final stage" vs "Flat-Aux at the final stage" (i.e., none).
- The primary-contrast equation is followed by a stray "." on its own line and then "at 25% budget".
- `CE\! -\!B` produces "CE - B" with odd spacing; use `\text{CE-B}`.
- The decision rule ("below one percentage point is not a strong positive headline") gives no minimum detectable effect or power calculation. See Section 4.

## 4. Statistics and power

- With five seeds, the seed SD of overall CIFAR-100 accuracy for ResNet-18 is typically a few tenths of a point, and larger on a hard subset. Partition-to-partition SD is unknown and probably larger. Estimate both in a pilot (e.g., 3 seeds × 5 random partitions at the 25% budget) and size the study for the 1-point effect the paper calls meaningful.
- Use a mixed model with partition and seed as random effects, or the rank test from C2, for H1. Fit the budget × method interaction for H3 with the same random-effects structure. "Paired training runs" should mean shared data subsets and initialization seeds across methods. Say so.
- There are three hypotheses, two budgets per experiment and several outcomes. Name one primary outcome per hypothesis (`C` for H1, `S` for H2, the interaction on overall accuracy for H3) and apply Holm correction across them. Everything else is exploratory.

## 5. Related work the paper needs

The manuscript has no references. At minimum:

- **Deep supervision:** Deeply-Supervised Nets (Lee et al., AISTATS 2015). This is the Flat-Aux condition.
- **Coarse-to-fine heads at intermediate layers:** B-CNN (Zhu & Bain, 2017) and HD-CNN (Yan et al., ICCV 2015). B-CNN is close to True-Hier's design.
- **Hierarchy-aware losses and mistake severity:** Making Better Mistakes (Bertinetto et al., CVPR 2020). This is directly relevant to C5.
- **Hierarchical contrastive objectives:** Use All The Labels (Zhang et al., CVPR 2022).
- **Example difficulty and prediction depth:** Baldock et al. (NeurIPS 2021). This is the source of Eq. (3).
- **Data-driven hierarchies:** label embedding trees (Bengio et al., NeurIPS 2010) and confusion-based hierarchies. These are the background for Visual-Cluster and the exploratory confusion-derived variant.
- **Label errors in CIFAR test sets:** Northcutt et al. (NeurIPS 2021 Datasets & Benchmarks). This supports C8.

Check each citation against the publisher's page before adding it.

## 6. Recommended revision of the design

1. **Outcomes.** Factorize accuracy into `C` (coarse correctness of the fine prediction) and `S` (sibling discrimination). H1 tests `C`, H2 tests `S`, and overall accuracy is co-primary. Probes at supervised layers become manipulation checks. PD is computed on the final pathway, with a fixed probe reference set.
2. **Partitions as the unit of inference.** Use about 19 matched partitions per family, rank True-Hier among them, and use fewer seeds per partition.
3. **Similarity axis.** Generate partitions spanning within-group similarity `W`, fit the outcome against `W`, and report True-Hier's residual as the semantic effect.
4. **Budget hygiene.** Build Visual-Cluster from a budget-internal or self-supervised extractor, and tune learning rate and weight decay per budget. Use fixed or curve-reported λ with gradient-norm matching instead of selection on 500 validation images.
5. **H3 restated** as a prior × data-scale interaction, or run as a semi-supervised experiment with coarse labels on extra images.
6. **Scope.** Retitle to CIFAR-100, or add one taxonomy-rich fine-grained dataset.
7. **Missing sections.** Add an introduction, related work, references and a full recipe table.

## 7. Suggested order of work (cheap checks first)

1. Compute `W(G)` for the true hierarchy, Random and Visual partitions, plus the ARI between True-Hier and the Visual partitions. Minutes, CPU only. If True-Hier and Visual-Cluster overlap heavily, the key contrast has little room.
2. From one CE model, compute `C` and `S` at 25% and 100% budget. This sets the baseline and shows how much room each factor has.
3. Pilot at the 25% budget: 3 seeds × 5 random partitions × {CE, True-Hier}. This estimates seed vs partition variance for the power calculation.
4. Fix the recipe and λ policy, then run Experiments A–C with the partition-rank design.
