# labelbench

labelbench evaluates classifiers against a gold standard and shows every
result in a self-contained HTML report that domain experts can browse row by
row. The classifier is interchangeable, so a hand-written rule set and an LLM
are run, evaluated and compared in exactly the same way, which makes it
possible to see within minutes whether a change to a classifier helped and
on which items it did not.

> **Status: version 0.2.** Projects, tasks and classifiers are created from
> templates, and a built-in LLM classifier works with OpenAI, Anthropic,
> OpenAI-compatible servers and any provider you connect yourself. Planned are
> tasks with several labels per item (`type: multi`), agreement between
> several annotators and an adapter for classifiers in other languages.
> [docs/design.md](docs/design.md) contains the architecture.

## Contents

- [How it works](#how-it-works)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [The project folders](#the-project-folders)
- [Creating your own task](#creating-your-own-task)
- [Checking a task](#checking-a-task)
- [Classifiers from templates](#classifiers-from-templates)
- [The LLM classifier](#the-llm-classifier)
- [Writing a classifier](#writing-a-classifier)
- [Commands](#commands)
- [The report](#the-report)
- [Working with dev and test](#working-with-dev-and-test)
- [Hierarchical labels](#hierarchical-labels)
- [All settings in task.yaml](#all-settings-in-taskyaml)
- [Reference](#reference)

## How it works

```
 task                     classifier                  run
 ─────────────            ─────────────────           ─────────────────────
 gold.csv      ──items──▶ rules, LLM, ...  ──────────▶ predictions.csv
 labels.csv               (from a config file)        metrics.json
 task.yaml                                            report.html
       └──────────────── gold labels ────────────────▶ (evaluation)
```

A **task** contains the items to classify together with their correct labels
and the list of allowed labels. A **classifier** sees only the features of
each item, never the correct label, and returns a prediction. A **run** stores
the predictions, compares them with the gold labels and writes metrics and a
report into a new directory that is never overwritten afterwards.

## Installation

labelbench requires Python 3.12 or newer. Create a folder for your project
with its own virtual environment and install labelbench from GitHub:

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install "labelbench[all] @ git+https://github.com/winf-hsos/labelbench"
```

On Linux and macOS the interpreter is `.venv/bin/python`. The extra `[all]`
installs the clients for OpenAI and Anthropic; use `[openai]` or
`[anthropic]` for only one of them, or no extra for rule-based classifiers
only. Installing makes the `labelbench` command available inside the
environment; activate it, or call `.venv/Scripts/labelbench` directly.

## Quickstart

**1. Create a project** with an example task, a rule-based classifier and,
optionally, an LLM classifier:

```bash
labelbench init my-project --llm openai
```

The example task routes 20 customer support messages to one of six topics.
Its files show the format of every file labelbench needs, so the quickest way
to learn the format is to look at them.

**2. Check the task and run the rule-based classifier:**

```bash
cd my-project
```

```bash
labelbench check --task tasks/example
```

```bash
labelbench run --task tasks/example --clf configs/keywords-v1.yaml
```

labelbench classifies all `dev` items, prints the main metrics and writes a
new run directory such as `runs/20261009-101500_example_keywords-v1/`. Open
the `report.html` inside it in any browser.

**3. Run the LLM classifier.** It needs an API key in the environment
variable named in its config, here `OPENAI_API_KEY`:

```bash
labelbench run --task tasks/example --clf configs/llm-v1.yaml
```

**4. Compare the two runs** item by item:

```bash
labelbench compare runs/<rules-run> runs/<llm-run>
```

The comparison lists the items that became correct, those that became wrong,
and whether the difference is larger than chance would explain.

## The project folders

```
my-project/
├── tasks/              one folder per task          ─┐
│   └── example/        task.yaml, labels.csv,         │ what is classified
│                       gold.csv                      ─┘
├── classifiers/        your Python code              ─┐
├── prompts/            prompt templates               │ how it is classified
├── configs/            one file per variant          ─┘
├── runs/               written by labelbench           the results
└── .labelbench-cache/  cached LLM answers              never edited
```

A run always combines **one task** with **one config**. The config names the
classifier, which is either your code in `classifiers/` or the built-in LLM
classifier with its prompts in `prompts/`. labelbench writes each run into a
new folder in `runs/`, together with copies of everything it was based on.

Never change a prompt or config that was used in a run. Copy it to a new
version (`llm-v2.yaml`, `llm-v2-system.md`) and change the copy, so that every
run stays reproducible and two versions can be compared with `compare`. Run
all commands from the project folder, because classifiers and prompts are
found relative to it.

## Creating your own task

```bash
labelbench new task tickets
```

This creates `tasks/tickets/` with three files:

| File | Content |
|---|---|
| `task.yaml` | names the columns for id, features and label; every optional setting is listed as a comment |
| `labels.csv` | one row per allowed label in column `label`, optionally a `description` and coarser levels |
| `gold.csv` | one row per item with id, feature columns and the correct label |

All CSV files are UTF-8 and comma-separated. Text fields may contain commas,
quotes and line breaks as long as they are quoted correctly, which every
spreadsheet program and CSV library does automatically. The full reference of
`task.yaml` is at the [end of this README](#all-settings-in-taskyaml).

## Checking a task

Gold standards that were built by hand almost always contain inconsistencies,
and some of them distort every evaluation without being visible in the
numbers. `labelbench check` looks for them before any classifier runs:

```bash
labelbench check --task tasks/tickets
```

Each finding has one of three severities. **Errors** make an evaluation
unreliable, and `labelbench run` refuses to start until they are fixed.
**Warnings** allow an evaluation but may distort it; they also appear in the
"Data checks" tab of every report. **Notes** are for information only.

| Check | Severity | What it finds |
|---|---|---|
| `unknown_gold_label` | error | gold labels missing from `labels.csv`, with the closest existing labels as suggestion and a hint when only surrounding whitespace differs |
| `duplicate_id`, `duplicate_label` | error | ids or labels that occur more than once |
| `empty_gold_label`, `empty_label` | error | items without gold label, rows in `labels.csv` without label |
| `bad_weight` | error | weights that are not non-negative numbers |
| `similar_labels` | warning | nearly identical labels, e.g. "Lupinen Bolognese" and "Lupinen-Bolognese", which penalise a classifier for picking the "wrong" twin |
| `conflicting_duplicates` | warning | items with identical features but different gold labels, which no classifier can all get right |
| `level_gold_mismatch` | warning | items whose own value on a level (see `level_gold`) differs from the value of their gold label |
| `label_whitespace`, `unknown_split`, `empty_level_value`, `unknown_level_value` | warning | formal problems in labels, splits and levels |
| `unused_labels` | note | labels that never occur in the gold standard, so their quality cannot be measured |
| `singleton_classes` | note | labels with a single gold example |
| `repeated_items`, `empty_features` | note | repeated inputs and items without any feature text |

`--all` lists up to 25 cases per finding instead of five.

## Classifiers from templates

`labelbench templates` lists what is available. A new classifier is created
with `labelbench new classifier NAME --template ...`; with `--task`, the
template is pre-filled with that task's labels and feature columns.

**Rule-based**: a Python class that assigns the first label whose keywords
occur in the text, with the keywords in the config. Extend the class with any
rules you need. This command creates `classifiers/rules.py` and
`configs/rules-v1.yaml`:

```bash
labelbench new classifier rules --template rules --task tasks/tickets
```

**LLM**: the built-in LLM classifier with a provider of your choice. This
command creates `configs/claude-v1.yaml`, `prompts/claude-v1-system.md` and
`prompts/claude-v1-item.md`:

```bash
labelbench new classifier claude --provider anthropic --task tasks/tickets
```

`--provider` implies `--template llm`. `--model` sets the model; without it the template uses the provider's default
shown by `labelbench templates`.

## The LLM classifier

The built-in classifier `llm` sends one request per item, restricts the
answer to the labels in `labels.csv` and caches every answer. Its config
looks like this:

```yaml
classifier: llm
params:
  provider: anthropic                       # see the table below
  model: claude-opus-5-5                    # any model name of the provider
  system_prompt: prompts/claude-v1-system.md
  prompt: prompts/claude-v1-item.md
  label_line: "- {label}: {description}"    # how one label appears in {{labels}}
  reasoning: true                           # ask for a short justification
  max_workers: 8                            # parallel requests
  options:                                  # passed to the provider
    effort: low
```

| Provider | For | Important options |
|---|---|---|
| `openai` | OpenAI models via the Responses API | `api_key_env` (default `OPENAI_API_KEY`), `reasoning_effort`, `max_output_tokens` |
| `anthropic` | Claude models via the Messages API | `effort`, `max_tokens`, `fallbacks`, `api_key_env` (by default the SDK finds `ANTHROPIC_API_KEY` itself) |
| `openai-compatible` | any server with an OpenAI-style API, e.g. a local model server | `base_url` (required), `api_key_env`, `json_schema: false` if the server lacks structured output |
| `module:function` | any other provider, through your own function | everything under `options` is passed to the function |

`--provider custom` creates such a function in `classifiers/` with
instructions. It receives the rendered prompts, the JSON schema and the model
name, and returns the model's answer as JSON text.

**Prompts** are two templates. The system prompt holds everything that is the
same for every item, typically the instructions and the label list, and the
item prompt holds the item's features. Providers can cache the system part,
which makes long label lists affordable. Placeholders use double braces:
`{{labels}}` inserts the label list, rendered line by line with `label_line`,
and `{{column}}` inserts a feature column of the item. labelbench refuses to
run if a prompt uses a placeholder that is neither.

**Answers** are JSON with a `label` and, unless `reasoning: false`, a short
`reasoning`. Where the provider supports it, the JSON schema restricts `label`
to the labels in `labels.csv`, so the model cannot invent one. The report shows
the reasoning next to each item, and the metadata records tokens, latency and
the model that actually answered.

**Caching**: every answer is stored in `.labelbench-cache/` under a hash of
provider, model, options and both rendered prompts. Repeating a run costs
nothing, and after a prompt change only the affected requests are sent again.
API keys are only ever read from environment variables, never from configs.

With `fallbacks: default`, Anthropic re-runs a request that a model's safety
classifier declined on a fallback model. The model that answered is recorded
per item, so such cases remain visible in the report.

## Writing a classifier

A classifier is a Python class with a `predict` method. It receives a list of
items, each a dictionary with the `id` and the feature columns as strings,
and returns one `Prediction` per item in the same order. The class created by
the `rules` template is a good starting point; in short:

```python
from labelbench import Prediction


class KeywordClassifier:
    def __init__(self, keywords: dict[str, list[str]], fallback: str | None = None):
        self.keywords = keywords
        self.fallback = fallback

    def predict(self, items: list[dict]) -> list[Prediction]:
        predictions = []
        for item in items:
            text = " ".join(v for k, v in item.items() if k != "id").lower()
            hits = [label for label, words in self.keywords.items()
                    if any(w in text for w in words)]
            predictions.append(Prediction(label=hits[0] if hits else self.fallback))
        return predictions
```

The config names the class as `module:Class` and passes everything under
`params` to the constructor, so one class can be evaluated with different
parameters without touching the code:

```yaml
classifier: classifiers.rules:KeywordClassifier
params:
  fallback: null
  keywords:
    refund_request: [refund, money back]
```

`Prediction` has four fields, of which only `label` is required:

| Field | Content |
|---|---|
| `label` | the predicted label, a list of labels for `type: multi`, or `None` if the classifier has no answer |
| `candidates` | further labels in ranked order, used for top-k metrics |
| `raw` | the raw output, e.g. an LLM response including its reasoning, shown in the report |
| `meta` | a dictionary with anything else worth keeping, such as tokens or runtime |

A label that is not listed in `labels.csv` is counted as the class
`__invalid__` instead of being ignored, so a classifier cannot look better by
answering nonsense on hard cases. If your classifier uses its own label names,
translate them with a `label_map` in the config instead of changing the
classifier:

```yaml
classifier: classifiers.rules:KeywordClassifier
label_map: {veggie: vegetarian}
params: {...}
```

If the class has a method `prepare(task)`, labelbench calls it once before
`predict`. The argument tells the classifier the task's name, its feature
columns, its levels and the full content of `labels.csv` as `task.labels`,
but never the gold labels.

For large jobs, the config may set `batch_size`; labelbench then calls
`predict` with chunks of that size and reports progress after each one. For
expensive calls of your own, `labelbench.JsonCache` offers the same cache the
LLM classifier uses. If a parameter of the config is the path of an existing
file, such as a prompt template, labelbench records its checksum in
`provenance.json`, so every run documents exactly which files it used.

## Commands

| Command | Purpose |
|---|---|
| `labelbench init [DIR] [--llm PROVIDER] [--model M]` | create a project with an example task, a rule-based and optionally an LLM classifier |
| `labelbench new task NAME` | create an empty task in `tasks/NAME` |
| `labelbench new classifier NAME --template rules [--task T]` | create a rule-based classifier, pre-filled from task T |
| `labelbench new classifier NAME --provider P [--model M] [--task T]` | create an LLM classifier for provider P (implies `--template llm`) |
| `labelbench templates` | list classifier templates, LLM providers and their default models |
| `labelbench check --task T` | check task T for consistency problems |
| `labelbench run --task T --clf C` | classify the `dev` items of task T (all items if it has no splits) with classifier C and evaluate |
| `labelbench run ... --split test` | evaluate on the held-out `test` items (logged) |
| `labelbench compare RUN_A RUN_B` | compare two runs item by item and write a comparison report |
| `labelbench report RUN` | recompute `metrics.json` and `report.html` of an existing run |
| `labelbench predict --task T --items F --clf C --out DIR` | classify unlabelled items in F with the label list of task T, e.g. a full dataset |

Each run directory is self-contained, so `report` and `compare` work even
if the original task files have changed since:

| File | Content |
|---|---|
| `predictions.csv` | one row per item with the predicted label, candidates, raw output and metadata |
| `metrics.json` | all metrics in machine-readable form |
| `report.html` | the report, a single file that needs no server |
| `task.yaml`, `labels.csv`, `gold.csv` | copies of the task, with `gold.csv` reduced to the evaluated items |
| `checks.json` | the findings of `labelbench check` on the full task at the time of the run |
| `config.yaml` | the classifier configuration exactly as used |
| `provenance.json` | checksums of task files, classifier config and prompt files, plus date, split and runtime |

## The report

`report.html` is meant for readers without a background in statistics, so
each number carries a short explanation of what it means.

- **Overview** shows accuracy and further metrics with confidence intervals,
  together with which gold standard, label list and classifier were used.
- **Classes** lists precision, recall and the number of examples per label,
  and the label pairs that were confused most often.
- **Items** shows every item, not only the errors, with its features, the
  correct label, the predicted label and the raw classifier output. It can be
  filtered by correct or wrong, by label and by free text.
- **Data checks** lists the findings of `labelbench check`, so readers see
  which problems of the gold standard may affect the numbers.
- **Run details** shows the classifier configuration and the provenance of
  the run.
- **Comparison** reports, created by `compare`, show the accuracy of both
  runs per level and list the items that changed between them.

Reviewers often notice that the gold label itself is wrong. Such rows can be
marked as "gold label doubtful" in the item details and exported as CSV or as
an Excel file, so the gold standard can be corrected afterwards. The Excel
file has a frozen header, a filter on every column and wrapped feature texts;
the report writes it itself, without any library or network access. The marks
are stored in the reviewer's browser only, which is why the export exists.

The language of the report is set with `language` in `task.yaml`; English
and German are available.

The report follows the light or dark setting of the viewer's system.
Appending `?theme=dark` or `?theme=light` to its address overrides that, for
example when the report is embedded in a dark presentation.

## Working with dev and test

If you improve a classifier many times against the same items, it ends up
fitted to exactly those items, and its measured quality is too optimistic.
labelbench therefore distinguishes two parts of the gold standard, marked in
the `split` column:

- `dev` items are used while developing, and `labelbench run` evaluates them
  by default.
- `test` items are only evaluated with `--split test`, ideally once for the
  final result. Each such evaluation is logged in `runs/test_access.log`.

Without a `split` column, all items are evaluated.

## Hierarchical labels

With many labels, a mistake between two similar labels matters less than one
between unrelated labels. Additional columns in `labels.csv` assign each
label to a coarser group, and labelbench evaluates every level separately:

```csv
label,description,group,diet
spaghetti_bolognese,"Spaghetti with minced meat sauce",pasta,meat
penne_arrabbiata,"Penne in spicy tomato sauce",pasta,vegan
fish_and_chips,"Breaded fish with fries",fish_dish,fish
```

```yaml
# task.yaml
levels: [group, diet]
```

The report then shows, for example, that a classifier picks the exact dish in
72 % of the cases but the right dietary category in 95 %.

By default, the gold value of an item on a level is derived from its gold
label. If annotators assigned the coarser value separately for each item, for
instance because a dish was vegetarian although its standard dish is vegan,
name that column in `level_gold`. Evaluation on that level then uses the
item's own value, and `labelbench check` lists every item where the two
disagree:

```yaml
levels: [diet]
level_gold: {diet: diet_of_item}   # column in gold.csv
```

## All settings in task.yaml

| Key | Required | Meaning |
|---|---|---|
| `name` | yes | name of the task, used in run directory names |
| `type` | yes | `single`; `multi` is planned |
| `id`, `label` | yes | columns with the item id and the gold label |
| `features` | yes | columns the classifier may see; the first is shown as the item's title in the report |
| `labels` | yes | file with the label list, relative to the task directory |
| `gold` | no | file with the gold standard, default `gold.csv` |
| `description` | no | one or two sentences shown at the top of every report |
| `levels` | no | columns of `labels.csv` that hold coarser levels |
| `level_gold` | no | per-item gold values for levels, `{level: column}` |
| `split` | no | column with `dev` or `test` |
| `weight` | no | column with a non-negative weight per item, e.g. how often a meal occurs in the full dataset |
| `show` | no | further columns of `gold.csv` that the report displays but the classifier never sees |
| `language` | no | language of the report, `en` (default) or `de` |

Comparing several annotators with each other (`annotators`) is planned.

## Reference

- [docs/formats.md](docs/formats.md) specifies every file format exactly.
- [docs/design.md](docs/design.md) explains the architecture and design
  decisions.

## License

labelbench is released under the [MIT License](LICENSE).
