# next-action-sft

Fine-tuning a small open model to decide what a policy-bound customer-service agent does
next, and measuring whether it actually helped.

A voice or chat agent that works under rules (verify identity before touching an
account, only offer a refund when policy allows) has to make one decision over and over:
given the conversation so far, which action now, and with what values. This project
trains Qwen2.5-1.5B-Instruct to make that decision on real human conversations, and
compares it with the same model given a detailed prompt and worked examples instead.

## The task

At every point where a human agent pressed an action button, predict the button and the
values recorded for it, from everything said and done before it:

```
customer: Hi! I need to return an item, can you help me with that?
agent:    sure, may I have your name please?
customer: Crystal Minh
                                    → {"action": "pull-up-account", "values": ["crystal minh"]}
...
customer: Username: cminh730
customer: cminh730@email.com
customer: Order ID: 3348917502
                                    → {"action": "validate-purchase",
                                       "values": ["cminh730", "cminh730@email.com", "3348917502"]}
```

The action is the policy half (30 possible, and the order matters). The values are
mostly extraction (copied from what the customer said, or chosen from a small set), with
some the agent picked from information the model never sees.
Both halves have to be right: **joint accuracy** is the headline number.

## Data

[ABCD](https://github.com/asappresearch/abcd) (Chen et al., NAACL 2021): 10,042
human-to-human customer service conversations, MIT licensed. People were paired live as
customer and agent, and agents followed a written company policy while clicking action
buttons. That makes it real human behaviour, but role-played with fictional customers.
It is not production call traffic.

One example per action, using ABCD's own train/dev/test split:

| | Conversations | Examples |
|---|---|---|
| train | 8,034 | 29,190 (8,000 used) |
| dev | 1,004 | 3,684 |
| test | 1,004 | 3,608 |

Measured before any model ran:

- **Floor.** Always predicting the most common action, `pull-up-account`, is right 19.7% of
  the time on test.
- **Extractable.** In 84.5% of test examples, every gold value appears in the
  conversation. The rest need something the human agent read off a dashboard, such as an
  FAQ article id or a refund amount. This is not a hard ceiling: some of those values can
  be learned (which FAQ article answers which question), and the check is lenient for
  very short values.
- **Label noise.** The values were typed by people. For actions with a fixed set of valid
  options, some gold labels fall outside it: 9.9% for `notify-team`, 6.6% for
  `shipping-status`, 1.3% for `membership` ("sliver", "order recieved").

Two more properties of the data were found only after the results, and are reported with
them below: customers repeat across splits, and some gold labels differ from what the
agent actually logged.

## Method

- **Model:** Qwen2.5-1.5B-Instruct (Apache-2.0), trained and run locally on an Apple M4
  with [MLX](https://github.com/ml-explore/mlx-lm).
- **Fine-tuning:** LoRA, rank 16, on the top 16 of 28 layers, 10.5M trainable parameters
  (0.68% of the model). One epoch over 8,000 examples, effective batch 8. The loss is
  computed on the JSON answer only, so the model learns to decide rather than to
  reproduce conversations.
- **Baselines:** the same base model zero-shot and five-shot, given a full description of
  every action and its values, plus zero-shot with only the action names. Five-shot uses
  one fixed set of five random training examples. The fine-tuned model gets only the
  action names: having learned the actions from 8,000 labelled examples, it does not need
  the descriptions, and it was not also run with them.
- **Scoring:** greedy decoding. Output that is not valid JSON of the right shape scores
  zero. Intervals are 95% cluster bootstraps over conversations, because examples from
  one conversation are correlated. The improvement is measured as a paired difference on
  the same examples.

Every choice, and why, is written down in [docs/decisions.md](docs/decisions.md),
including the ones that changed partway through.

## Results

On ABCD's full test set, 3,608 examples from 1,004 conversations, scored once. The
checkpoint was chosen beforehand on a separate 500-example dev sample.

| System | Prompt | Joint | Action | Values | Valid JSON |
|---|---|---|---|---|---|
| Base, zero-shot | action names only | 5.8 (5.0–6.6) | 15.7 | 16.5 | 69.3 |
| Base, zero-shot | full descriptions | 9.4 (8.5–10.4) | 18.1 | 22.8 | 76.6 |
| Base, five-shot | full descriptions | 24.6 (23.1–26.0) | 34.9 | 38.2 | 98.9 |
| **Fine-tuned (LoRA)** | **action names only** | **71.3 (69.2–73.2)** | **81.3** | **75.5** | **99.8** |

Percentages; brackets are 95% cluster-bootstrap intervals over conversations. The tables
are copied from `nextaction.report` and `nextaction.analysis`, which build them from the
saved predictions. `tests/test_readme_numbers.py` recomputes each result figure in this
README from the result files and checks it in context: a table row as a whole row, a
figure in a sentence with the words around it. Any other number in the README must be on
a short list of constants such as the LoRA rank, or the test fails. The full version, with intervals on every
metric, is in [results/report-test.md](results/report-test.md).

**Fine-tuning beats the strongest baseline tried, five-shot, by 46.7 points of joint
accuracy** (paired 95% interval 44.6 to 48.7). On the same examples, the fine-tuned model
is right where five-shot is wrong 1,770 times, and the reverse 85 times. The descriptions
help the base model a little: 5.8% with action names only, 9.4% with descriptions.

What each step bought:

- **Five examples fixed the format, not the decision.** Valid JSON went from 76.6% to
  98.9%, but the base model still chose the right action only 34.9% of the time.
- **Untrained, the base model is worse than a constant.** Its zero-shot action accuracy,
  18.1%, is below the 19.7% from always answering `pull-up-account`.
- **Fine-tuning taught the policy.** Action accuracy rose to 81.3%, and value accuracy
  to 75.5%.
- **Dev and test agree.** 72.8% joint on the dev sample used to pick the checkpoint, 71.3%
  on test, which is consistent with the choice not overfitting to dev. The ranking of
  checkpoints on dev was itself noisy (62.6, 69.8, 66.8, 72.8, each about ±4).

### Measured after the results

Everything in this section was computed after the test scores were known.

**Customers repeat.** ABCD reuses a small pool of fictional customers. Of the 709 test
`pull-up-account` actions, only 22 have a value absent from the training sample, and at
most one is a new customer: 16 are account ids, 5 are misspellings of customers in
training, and 1 is a name not in the training sample. So part of what looks like
extracting the customer's name could be recall.

That makes it worth splitting test by whether each gold value appears among the answer
values of the 8,000 training examples (pooled across actions; joint accuracy, %):

| Test examples | n | Base, five-shot | Fine-tuned |
|---|---|---|---|
| Action takes no values | 1,076 | 21.7 | 79.1 |
| Every value seen in training | 1,679 | 27.6 | 72.1 |
| **Some value never seen in training** | **853** | **22.3** | **59.8** |
| Every value in the conversation | 3,050 | 28.1 | 78.1 |
| Some value not in the conversation | 558 | 5.2 | 33.9 |

The advantage holds on examples with a value never seen in training, by 37.5 points, so
the headline is not explained by memorisation, though recall may help. The split is not
clean: 70% of the never-seen group is identity and purchase checks (`verify-identity`,
`validate-purchase`), so it also differs from the rest in which actions it contains.

**Gold labels are not always what the agent logged.** For the 8 actions whose own system
message repeats the value entered ("Account has been pulled up for Crystal Minh."), the
value was read back out of the message and compared, whole, with the gold label. They
differ in 64 of 1,507 test examples. In 41 the agent logged nothing ("Account has been
pulled up for ."), and the gold label supplies the value. In 8 they nearly match, a typo or
a truncation on one side ("chloe zang" against "chloe zhang"). In 15 they are different
values ("michael kors jeans" against a logged purchase of a Tommy Hilfiger shirt).

### Where it is weak

Joint accuracy by action is in the full report. Worth knowing:

- **Five-shot beats the fine-tuned model on two actions**: `shipping-status` (66.3 against
  51.2, 86 examples) and `log-out-in` (76.1 against 68.5, 92 examples), and ties on two
  more. `shipping-status` is one of the five demonstrations, which likely explains that
  one; `log-out-in` is not. These per-action samples are small and their intervals wide.
- **At least one error is a label error.** In 1 test example the fine-tuned model
  predicted exactly the value the agent logged, and was scored wrong against a gold
  label that differs from it.
- **`select-faq` is the weakest at 39.2%**, as expected: the value is an FAQ article id
  the human chose from a knowledge base, and it never appears in the conversation.
- **Many of the most frequent action errors look like order, not meaning.** The single
  most common wrong answer is `ask-the-oracle`, a check the agent runs, predicted in
  place of what the human did first; another frequent one is pulling up the account
  where the human verified identity. Some may be valid alternative orders that
  exact-match scoring counts as wrong. Neither the share of errors that are ordering
  errors nor their validity has been measured, so this is a hypothesis, not a claim.

### Inference numerics

Batched generation is not bit-identical to generating one example at a time. On 200 dev
examples the evaluation configuration (batch 16 with a cached shared prompt) moved
base-model scores by 1.0 to 2.5 points against the one-at-a-time reference, with 180 of
200 outputs identical (`nextaction.numerics`, saved in
[results/numerics.json](results/numerics.json)). All systems ran in the same configuration,
and the rule set in advance, rerun in the reference configuration if the gain were under
3 points, was not triggered. Details in [docs/decisions.md](docs/decisions.md).

## Limitations

- **Role-played, not production.** ABCD is real people, but fictional customers and a
  text chat, not phone audio with speech-recognition errors.
- **The baselines are not exhaustive.** Five-shot uses one fixed random set of
  demonstrations covering 5 of 30 actions. Demonstrations retrieved per example, or a
  larger model, would be stronger baselines and were not tried.
- **Not compared with the ABCD paper's numbers.** Its metrics and formulation differ
  from this text-to-JSON setup (its repository trains BERT-style encoders on the
  delexicalised conversations), so the numbers are not directly comparable.
- **Exact-match scoring.** A different but valid order of actions counts as wrong, and so
  does a value that differs from a human typo in the gold label.
- **One run.** One seed and one 8,000-example sample, and training was not repeated, so
  how much a rerun would vary is unknown. The intervals cover sampling of test
  conversations, not variation between training runs.
- **SFT only.** No preference tuning (DPO) or reinforcement learning. The ordering errors
  above are a natural target for DPO: pairs of the human's next action against the
  model's out-of-order one.

## Reproduce

```bash
make setup
make data                    # ABCD from its GitHub repo (37 MB)
make prepare
make baseline                # the three base-model systems
make train                   # about 3 hours on an M4 with 24 GB
make select                  # score checkpoints on dev500, pick one (never on test)
make eval && make report
make analysis numerics       # memorisation, label quality, batching check
```

The LoRA adapter is not in the repository. Trained with mlx-lm 0.31.3; GPU training is
not guaranteed to be bit-for-bit reproducible, and this run was not repeated.

## Layout

| Path | |
|---|---|
| `src/nextaction/data.py` | ABCD conversations into one example per agent action |
| `src/nextaction/prompt.py` | The full and compact system prompts |
| `src/nextaction/evaluate.py` | Batched greedy evaluation with a shared-prefix cache |
| `src/nextaction/metrics.py` | Scoring, cluster bootstrap, paired differences, the extractable share |
| `src/nextaction/report.py` | Builds the results tables from saved predictions |
| `src/nextaction/analysis.py` | Memorisation, extraction and label-quality breakdowns |
| `src/nextaction/numerics.py` | Batched against one-at-a-time generation |
| `configs/lora.yaml` | The training run |
| `docs/decisions.md` | Every decision, why, and when it was made |
| `results/` | Summaries, the report, and every prediction behind them |
