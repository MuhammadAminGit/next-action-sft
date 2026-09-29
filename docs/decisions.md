# Decisions

Each choice made in this project and the reason for it. Entries were written as the work
happened, before the results they could affect were known, and each rule says at what
point in the run it was fixed. The repository's first commit came after the experiments,
so git history cannot confirm that order; the modification times of this file and of
`results/` are consistent with it.

Text added once the test results were known is marked: *Correction* where an entry was
wrong, *Added after the results* where detail was added. In both cases the original
wording is kept above it. The last section is written entirely after the results.

## Data

**ABCD, not synthetic transcripts.** The goal is post-training on real data. ABCD is
10,042 human-to-human customer service conversations where the agent must follow a
written policy, collected by pairing people as customer and agent live. It is real
human behaviour but role-played with fictional customers, not production traffic.
MIT licensed, with official train/dev/test splits.

**One example per agent action.** Input is everything said and done before the action;
target is `{"action", "values"}`. This is ABCD's Action State Tracking task as
text-to-JSON. Earlier actions stay in the input because the agent saw them and because
order is the policy (verify identity after pulling up the account, not before).

**8,000 of the 29,190 training examples, one epoch.** Enough to learn the task in about
an hour on an M4. Using all of them is the obvious next experiment, not a default.

*Correction:* training took about three hours (see the batch-size entry). The estimate
was off because the M4 runs a 1.5B model's backward pass slowly.

**Measured before any model ran** (`results/data_stats.json`):
- Floor: always predicting `pull-up-account` gets 19.7% action accuracy on test.
- Ceiling: 84.5% of test examples have every gold value present in the conversation.
  The rest need something the agent saw on a dashboard (an FAQ article id, a refund
  amount), so ~85% is near the practical maximum for values, not a failure to reach 100.
- Label noise: gold values outside the official options for notify-team (9.9%),
  shipping-status (6.6%), membership (1.3%). Humans typed "sliver" and "order recieved".

*Correction:* 84.5% is not a maximum. The fine-tuned model gets 33.9% joint on the
examples whose values are not in the conversation, because some (which FAQ answers which
question) are learnable, and the check is lenient for very short values. The README
calls it the extractable share.

## Model

**Qwen2.5-1.5B-Instruct**, Apache-2.0. Small enough to train locally and weak enough at
the task untrained (about 6-12% joint on dev samples) that fine-tuning has room to show
an effect. No thinking mode to strip.

**LoRA, rank 16, on 16 of 28 layers; loss on the answer only.** Only the JSON answer is
trained on, not the system prompt or the conversation, so the model learns to decide,
not to reproduce transcripts.

**Effective batch 8, reached as batch 2 with 4 steps of gradient accumulation.** Batch 8
ran out of GPU memory on the 24 GB M4 at the first step. Batch 4 with accumulation 2 ran
out at about step 350, once longer conversations arrived (peak memory climbed from 15.7
to 19.5 GB). Batch 2 with accumulation 4 peaks under 10 GB. The optimiser sees the same
effective batch and makes the same 1,000 updates, so the training is equivalent; it was
also slightly faster per example (0.84/s against 0.68/s). Checkpoints every 250 steps so
a crash cannot lose more than a few minutes.

**The built-in validation loss is too small to trust, so checkpoints are chosen on dev.**
`val_batches` was set when the batch was 8; after the switch to batch 2 it silently
covers only 20 examples. The curve read 2.175, 0.126, 0.074, 0.179 at steps 1, 500,
1000 and 1500. The rise at 1500 could be overfitting or noise on 20 examples; there is no
way to tell. Rule, fixed at step 1,900 before any checkpoint was evaluated: the
checkpoints at steps 1000, 2000, 3000 and 4000 are scored on a fixed random sample of 500
dev examples, and the one with the best joint accuracy is the model evaluated on test.

**Result of that rule: step 4000, the final checkpoint.** Joint accuracy on the 500 dev
examples: step 1000 62.6%, step 2000 69.8%, step 3000 66.8%, step 4000 72.8%. The
selected model is simply the end of training, so no mid-run checkpoint was cherry-picked.
The test set had not been touched by any system at this point.

*Added after the results:* the dev sample is drawn with seed 2026, now by
`nextaction.prepare`. The dip at step 3000 suggests the ranking between checkpoints, each
about ±4 points, is largely noise.

## Evaluation

**Three systems.** Base zero-shot, base five-shot, fine-tuned zero-shot. Five-shot is the
baseline that matters: beating only zero-shot would leave open that better prompting
could have done the same.

*Added after the results:* five-shot uses one fixed random set of five training examples
(seed 7), covering `pull-up-account`, `validate-purchase`, `search-membership`,
`shipping-status` and `promo-code`. Retrieved per-example demonstrations would be a
stronger baseline and were not tried.

**Two prompts, and the asymmetry favours the baselines.** The plan was one identical
prompt for every system. The first training run showed the full prompt, which describes
every action and its values, is 710 tokens: 82% of a median 864-token example, all of
it masked out of the loss. Training on it projected to seven hours or more on the M4,
mostly re-reading constant text.

So the fine-tuned model is trained and evaluated with a compact prompt (action names
and output format only; median example 325 tokens), while the baselines keep the full
descriptions. The fine-tuned model is told strictly less and has to learn what each
action means from data, so any win it shows is conservative. The base model is also
run on the compact prompt, which measures what the descriptions are worth rather than
assuming it.

*Correction:* the heading and "any win it shows is conservative" are not supported. The
fine-tuned model learned the actions from 8,000 labelled examples, which makes the
descriptions redundant for it, and it was never run with the full prompt. The accurate
claim is only that it does not need them.

**Prompt fix from dev, before any test run.** A dev sample showed `pull-up-account`
sometimes takes an account id, not a name. The action description was corrected.
Nothing is ever changed in response to test results.

**Joint accuracy is the headline.** A right action with a wrong order number is wrong.

**Cluster bootstrap over conversations** for every interval, because examples from one
conversation are correlated. **Paired differences** for the improvement.

## Inference numerics

**Batched generation is not bit-identical to generating one example at a time.** On 48
dev examples with the base model, batch 16 changed 6 outputs relative to batch 1, with
or without a cached prompt prefix; the prefix cache on its own changed 2. This is
floating-point: padded batches do the same arithmetic in a different order, and a model
that is nearly tied between two tokens flips.

On 200 dev examples, the fast configuration (batch 16, shared-prefix cache) against the
reference (batch 1, no cache): 180 of 200 outputs identical; metrics moved by +1.0 to
+2.5 points, joint +1.0 (paired 95% CI +0.0 to +2.5).

**Rule, fixed before any test result was seen:** every system is evaluated in the same
fast configuration, and this effect is reported as a known uncertainty. If the measured
fine-tuning gain on joint accuracy is under 3 points, the full comparison is rerun in the
reference configuration before any claim is made.

*Added after the results:* the measured gain was 46.7 points, so the rule did not
trigger. The 200-example check was first run by hand; `nextaction.numerics` reruns it and
saves `results/numerics.json`, which reproduces the numbers above exactly.

## After the results

Everything in this section was measured after the test scores were known. None of it
changed a reported headline number.

**Customers repeat across splits.** ABCD reuses a small pool of fictional customers. Of
709 test `pull-up-account` actions, 22 have a value absent from the training sample, and
at most one is a new customer: 16 are account ids, 5 are misspellings of customers in
training, and 1 is a name not in the training sample. `nextaction.analysis` therefore splits test by whether each gold
value appears among the training answers. On examples with a value never seen in
training, the fine-tuned model scores 59.8% joint against five-shot's 22.3%, so the gain
is not explained by memorisation, though recall may help. The split is confounded by
action: 70% of the never-seen group is `verify-identity` and `validate-purchase`.

**Gold labels are not always what the agent logged.** For the 8 actions whose system
message repeats the entered value, the value was read back out of the message template
and compared, whole, with the gold label. They differ in 64 of 1,507 test examples: in 41
the agent logged nothing, in 8 the two nearly match (a typo or truncation on one side),
and in 15 they are different values. In 1 of those 15 the fine-tuned model predicted
exactly the logged value and was scored wrong. Scores are reported against the gold
labels as given.
