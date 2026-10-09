# Design

labelbench evaluates classifiers against a gold standard. The classifier is
interchangeable: a rule-based script and an LLM are run, evaluated and
compared in exactly the same way. The goal is to see within minutes whether a
change to a classifier made things better or worse, and why.

The first application is the classification of German university canteen
meals (dietary category, mapping to ~200 standard dishes), but nothing in the
tool is specific to that domain.

## Three separate parts

**Task** — gold standard plus label schema, described in `docs/formats.md`.
The label schema is data, not code, so new tasks need no code changes.

**Classifier** — anything that turns items into predictions. In Python it
implements

```python
@dataclass
class Prediction:
    label: str | list[str] | None      # None = no valid answer
    candidates: list[str] | None = None
    raw: str = ""
    meta: dict = field(default_factory=dict)

class Classifier(Protocol):
    def predict(self, items: list[dict]) -> list[Prediction]: ...
```

The interface works on batches so an LLM classifier can parallelise. Which
classifier runs with which parameters is defined in a config file, so a
changed prompt is a new file rather than a code change:

```yaml
classifier: llm                        # built-in short name, or
# classifier: mypkg.rules:PaperRules   # any importable class
label_map: {}                          # classifier output -> task labels
params:                                # passed unchanged to the constructor
  model: claude-sonnet-5
  prompt: prompts/standard-dish-v3.md
  temperature: 0
```

Keys at the top level belong to labelbench, everything under `params`
belongs to the classifier, so the two can never collide. Custom classifiers
are referenced by import path and resolved relative to the working
directory, so project-specific code (such as a published rule set) stays in
the project that uses labelbench.

The LLM prompt is a template with placeholders in double braces:
`{{labels}}` expands to the label list with descriptions from `labels.csv`,
and `{{<feature>}}` inserts a feature column. Double braces are used because
prompts often contain JSON examples with single braces.

**Run** — an immutable directory with predictions, metrics, provenance and
the HTML report. The evaluator is a pure function of `gold.csv`,
`labels.csv` and `predictions.csv`.

## Command line

```
labelbench run     --task path/to/task --clf configs/llm-v3.yaml   # dev by default
labelbench run     --task path/to/task --clf configs/llm-v3.yaml --split test
labelbench compare runs/<run_a> runs/<run_b>
labelbench report  runs/<run>          # rebuild report.html from files
labelbench check   --task path/to/task
labelbench predict --task path/to/task --items all.csv --clf configs/llm-v3.yaml --out out/
```

`predict` classifies unlabelled items and writes only `predictions.csv` plus
provenance. It is the production path and uses the same classifier code and
config as `run`.

`compare` is the main tool for iteration. Besides metric differences it lists
the items that became correct and those that became wrong, and runs a
McNemar test on the paired results. It warns if the two runs used different
gold standards or label schemas.

## Metrics

Computed for the label itself and for every level in `levels`:

- accuracy, optionally frequency-weighted via `weight`
- macro-F1 over the classes present in the gold standard, reported with
  their number
- precision, recall, F1 and support per class
- Cohen's kappa
- top-k accuracy when classifiers return `candidates`
- share of `__invalid__` predictions
- inter-annotator agreement when `annotators` is set, as a reference line
- bootstrap confidence intervals for all headline numbers

With ~200 labels and a gold standard of a few hundred items, most classes
have very few examples. Per-class numbers are therefore always shown with
their support, and a full confusion matrix is only drawn for small label
sets. Large label sets get a ranked list of the most frequent confusion
pairs instead.

## HTML report

One self-contained file per run with the data embedded as JSON. It needs no
server, can be sent by e-mail and published via GitHub Pages. The audience
includes domain experts without a statistics background, so every metric
carries a one-sentence explanation.

1. **Overview** — headline numbers with confidence intervals per level, and
   which gold standard, label schema and configuration were used.
2. **Classes** — sortable, filterable table per label, plus confusion pairs
   that expand to their examples.
3. **Items** — every row, not only the errors, filterable by correct/wrong,
   gold label, predicted label and free text. Each row shows the features,
   the raw classifier output and the candidate list.
4. **Comparison** — only in reports produced by `compare`: items that
   changed from wrong to correct and vice versa.

Reviewers can flag rows as "gold label doubtful" and export the flagged rows
as CSV, because reviewing errors regularly uncovers errors in the gold
standard itself.

## Principles

- **Runs are immutable.** A run never overwrites an existing directory, and
  every number in a paper can be traced back to one run.
- **`predictions.csv` is the contract.** Evaluator and report never call a
  classifier, which is what makes classifiers in R or other languages
  possible later.
- **Iterate on `dev`, report on `test`.** Evaluating `test` requires an
  explicit flag and is logged.
- **Same code path for evaluation and production.** The bulk classification
  of the full dataset uses the same classifier and config that was
  evaluated, only without gold labels.
- **LLM responses are cached** by model, prompt, parameters and input, so a
  re-run only sends what actually changed.

## Consistency checks

Hand-made gold standards contain inconsistencies that distort evaluations
without showing up in the numbers, such as gold labels that are missing from
the label list, near-duplicate labels or identical inputs with different gold
labels. Loading a task therefore separates two kinds of problems.
Structural ones (a missing file, key or column) raise at once, because nothing
else can be checked without them. Content problems are collected by
`labelbench.checks` as findings with a severity, so `labelbench check` can
show all of them together. Errors block `run`, while warnings are stored in
`checks.json` and shown in every report, so readers of a result see which data
problems may affect it.

## Status

Version 0.2 implements task loading with checks, the runner, metrics with
levels, per-item level gold, weights, top-k and bootstrap intervals, `compare`
with the McNemar test, the HTML report in English and German, `predict`, the
built-in LLM classifier with providers for OpenAI, Anthropic,
OpenAI-compatible servers and custom functions, and project, task and
classifier templates (`init`, `new`, `templates`). Still open:

1. `type: multi` for several labels per item.
2. Agreement between several annotators as a reference line.
3. The adapter for classifiers in other languages via `items.csv` /
   `predictions.csv`.

## Templates and providers

`labelbench init` and `labelbench new` copy files from `templates/` inside
the package and fill in a few placeholders (`@@name@@`), never overwriting an
existing file. Templates contain no domain knowledge beyond the neutral
example task; with `--task`, labels and feature columns of an existing task
pre-fill them.

The built-in LLM classifier keeps provider differences in
`labelbench/providers/`, one module per API, each with a single `complete`
method that returns the model's JSON text. Prompt rendering, the JSON schema
that restricts the answer to the label list, caching, parallelism and parsing
are shared, so every provider is evaluated under identical conditions. API
keys are read from environment variables only.
