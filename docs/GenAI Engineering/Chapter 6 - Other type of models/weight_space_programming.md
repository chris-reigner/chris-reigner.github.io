# Weight-Space Programming

## What is it?

A trained network is usually a black box. We change its behaviour from the outside: more data, fine-tuning, a better prompt, guardrails. The weights are the output of training, and nobody edits them by hand.

Weight-space programming treats the weights as a program instead. You **read** them to understand what the network computes, and you **edit** them directly to change it. The edit comes from analysis, with no training loop involved: find the component responsible for a behaviour, check the hypothesis, then patch it.

A simple way to picture the difference:

- **Prompting** changes the input of the program.
- **Fine-tuning** recompiles the program on new data and hopes the bug goes away.
- **Weight-space programming** opens the binary, finds the faulty function and patches it.

## How it differs from what we already do

It belongs to a family of techniques that change weights with little or no training:

- **Model merging / task arithmetic**: add or average weight deltas between fine-tuned checkpoints. Generic, behaviour-agnostic.
- **Knowledge editing** (ROME, MEMIT): locate the MLP that stores a fact and rewrite it with a rank-one update. One fact at a time.
- **Directional ablation**: find the direction that carries a behaviour (refusal is the well-known case) and project it out of the weights.
- **Activation steering**: add or remove a direction at inference time with a hook. Same idea, but it lives in the serving code instead of the checkpoint.
- **LoRA**: also a low-rank `ΔW`, but learned by gradient descent on a dataset.

What is new is that one loop handles very different goals: identifying what a network was trained on, writing a small policy from scratch, or repairing broken reasoning in an LLM.

The unit that makes this work is usually a **direction in the residual stream**. Layers read from and write to this shared space, and a behaviour like "keep repeating yourself" often lines up with one direction written by a few layers. If you know that direction `u` and the matrix that writes it, removing it is a rank-one edit:

    W_out_new = W_out - u (uᵀ W_out)

The rest of the model is untouched.

## Modifying a small model with a frontier one

The experiment that made this concrete comes from MIT CSAIL (Takehi, Teoh, Zhang, Khattab, November 2026). A frontier model gets the weights of a smaller network in a Python sandbox, plus a goal in plain English. Training loops, gradient descent and search are forbidden; it can only inspect, reason, edit and run a limited number of forward passes.

The headline case is Qwen3.5-0.8B. In thinking mode it hit the 32,768-token limit on 93% of math answers because it kept repeating itself. The frontier model traced the loop to one direction written by the MLPs of layers 15 and 19, and subtracted it from 64 output columns in each layer.

| | Before | After |
| --- | --- | --- |
| Weights modified | - | 131,072 of 0.8B (0.016%) |
| MATH-500 | 7.9% | 48.1% |
| GSM8K | 13.8% | 67.1% |
| IFEval | 21.0% | 62.3% |
| Answers hitting the token limit | 93% | 33% |

The point I keep coming back to is what was **not** needed. There was no fine-tuning dataset, no reward model and no GPU training run. A fine-tuning job would probably have fixed the looping too, but you would get a full new checkpoint and no explanation of why it worked. Here the fix is a small delta you can review and revert.

The same setup also worked on other networks: naming which game a small policy plays from its weights alone, writing MinAtar policies that beat PPO trained for 100M steps, and fixing a failing car-driving policy by editing 3.5% of its weights.

## Does that mean language models are explainable?

Well, not really... At least if you compare against explainability field in machine learining.

A successful edit is strong evidence for an explanation. If you claim "this direction causes the looping", remove it, and the looping drops from 93% to 33% while other benchmarks go up, the claim has passed a causal test. Most interpretability work stops at correlation; this closes the loop.

But the explanation stays local. It tells you why the model loops, not how it solves a math problem. It's worth noticing:

- **It works when behaviours are localised.** Looping or refusal sit in a few directions. Reasoning skill or world knowledge is spread across the network and nobody can patch it this way.
- **Reading is still hard.** The same frontier agent could not tell which model in a pair was sycophantic from the weights alone. It needed attention-analysis tools to get there.
- **The explainer is itself a black box.** When a frontier LLM does the reading, you get its rationale in plain English, but that rationale comes from a model we can't fully inspect either.

So I would say weight-space programming makes specific behaviours **testable**, which is a step toward explainability without being the whole thing.

## Using it to audit, fix and validate smaller models

This is where I see the practical value for teams running small open-weight models.

### Audit

Inspect the weights for a known behaviour before running any benchmark. The read experiments (training language, game played, presence of copying heads) show that a model's weights say a lot about how it was trained. Applied to procurement, that means checking a third-party checkpoint for a suspicious direction (sycophancy, a backdoor trigger, a refusal pattern removed upstream) before you even deploy it.

### Fix

Patch a well-defined failure mode with a rank-one delta. Good candidates are behaviours you can reproduce and describe in a sentence: endless loops, a broken output format, over-refusal on a class of prompts. The patch is small enough to store next to the checkpoint and review like a code change.

### Validate

Use edits as experiments. If your team believes "component X causes behaviour Y", remove X and measure. A claim that survives the edit can go into the model card; one that doesn't is just a hypothesis.

## Do you even need a frontier model?

Fixing a 0.8B model with a model that costs dollars per million tokens sounds backwards. AHA (Khamis, Tutunov, Bou Ammar, November 2026) answers this directly: you don't.

AHA rebuilds the looping fix from the small model's **own outputs**:

1. **Direction**: sample two continuations from the same prefix, one that loops and one that reaches an answer. Since the prompt is identical, the difference between their residual streams isolates the "loop" direction.
2. **Layers**: pick them with a small linear program scored on held-out data (building on Nathan Barry's linear-programming formulation).
3. **Neurons**: pick the few output columns to edit with sparse optimisation.

The edit has the same shape as before, a rank-one `ΔW = u vᵀ`, and the results match the agent-based approach:

| Benchmark | Model | Unedited | AHA |
| --- | --- | --- | --- |
| GSM8K | Qwen3.5-0.8B | 14.0 | 71.2 |
| MATH-500 | Qwen3.5-0.8B | 9.8 | 51.8 |
| IFEval | Qwen3.5-0.8B | 19.6 | 62.7 |
| GSM8K | Qwen3.5-2B | 75.8 | 88.4 |

An edit takes about 22 minutes on a single RTX 4090, roughly $0.30. The whole study cost $7.50 in GPU time.

!!! note "Baselines differ"
    The MIT and AHA papers report different unedited scores (MATH-500 7.9 vs 9.8) because their evaluation setups differ. Compare the gains within each paper rather than absolute scores across papers.

My takeaway: the frontier model is useful for **discovering** what to edit, and once the pattern is known it can be turned into a cheap, deterministic algorithm. I expect that cycle to repeat: an agent explores a new kind of edit, then someone distils it into a script.

## Limitations

- **It only surfaces what the model already knows.** Gemma 3 270M, which rarely loops, could not be improved. The edited 0.8B model still trails the unedited 2B on MATH-500 (51.8 vs 69.6).
- **Gains shrink on harder problems.** AHA gains about 60 points on level-1 MATH problems and about 20 on level 5. The fix helps most where the model already knew the answer and failed to finish.
- **Small models only, for now.** Evidence covers sub-2B models from a couple of families. Reading a 70B model through an LLM's context window is not practical yet.
- **Small edit, global effect.** A rank-one change applies to every input, so side effects can show up far from the target behaviour.
- **Few head-to-head comparisons.** Both works show feasibility. Neither compares systematically against fine-tuning or steering at equal cost.
- **Determinism.** AHA results vary slightly across GPU implementations.
- **Safety cuts both ways.** If anyone with the weights can remove a behaviour for $0.30, safety training on open-weight models becomes easier to undo. Earlier work on the refusal direction already showed this.

## Next steps

Where I expect the field to move, and what I will watch:

- **Auditing tools**: scanners that flag known directions (sycophancy, refusal removal, backdoors) in a checkpoint, the way we scan dependencies for CVEs.
- **Better reading at scale**: sparse autoencoders and circuit-discovery tools feeding agents, so larger models become tractable.
- **More behaviours beyond looping**: format errors, over-refusal and language switching are natural next targets for the matched-pair recipe.
- **Transfer across models**: whether a direction found in one model family can be mapped to another.
- **Comparisons with fine-tuning**: the same failure fixed by LoRA, steering and a weight edit, with cost, side effects and regression results side by side.

The Weight Space Learning survey (Han et al., 2026) is a good map of the academic side: it covers weight symmetries, models that take other models' weights as input, and generating weights with hypernetworks or diffusion. Weight-space programming uses the same objects but swaps the learned meta-model for reasoning.

## Sources

- Takehi, Teoh, Zhang, Khattab, *Sparks of Weight-Space Programming* (MIT CSAIL, 2026): <https://rikiyat.github.io/blog/weight-space-programming/>
- Khamis, Tutunov, Bou Ammar, *Who Needs an LLM to Program an LLM? Introducing AHA!* (2026): <https://haithamb.github.io/aha/>
- Han et al., *A Survey of Weight Space Learning: Understanding, Representation, and Generation* (2026): <https://arxiv.org/abs/2603.10090>
- ROME, locating and editing factual associations: <https://arxiv.org/abs/2202.05262>
- Refusal in LLMs is mediated by a single direction: <https://arxiv.org/abs/2406.11717>
- Representation engineering: <https://arxiv.org/abs/2310.01405>
- Task arithmetic: <https://arxiv.org/abs/2212.04089>
