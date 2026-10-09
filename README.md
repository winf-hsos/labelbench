# labelbench

labelbench evaluates classifiers against a gold standard and shows every
result in a self-contained HTML report that domain experts can browse row by
row. The classifier is interchangeable, so a hand-written rule set and an LLM
are run, evaluated and compared in exactly the same way, which makes it
possible to see within minutes whether a change to a classifier helped and
on which items it did not.

> **Status: version 0.1.** `check`, `run`, `compare`, `report` and `predict`
> work for single-label tasks. Planned but not yet implemented are tasks with
> several labels per item (`type: multi`), a built-in LLM classifier and an
> adapter for classifiers in other languages; sections that describe them say
> so. [docs/design.md](docs/design.md) contains the architecture.

## Contents

- [How it works](#how-it-works)
- [Installation](#installation)
- [What you need](#what-you-need)
- [Quickstart](#quickstart)
- [Checking a task](#checking-a-task)
- [Writing a classifier](#writing-a-classifier)
- [Using an LLM as classifier](#using-an-llm-as-classifier)
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

labelbench requires Python 3.12 or newer and is not yet published on PyPI.
Install it into the virtual environment of the project that uses it:

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install -e path/to/labelbench
```

On Linux and macOS the interpreter is `.venv/bin/python`. Installing makes
the `labelbench` command available inside the environment. Libraries your
own classifiers need, such as an LLM client, are installed the same way.

## What you need

labelbench is used from your own project directory. A typical layout:

```
my-project/
├── tasks/
│   └── diet/
│       ├── task.yaml      required: describes the task
│       ├── labels.csv     required: the allowed labels
│       └── gold.csv       required: items with their correct label
├── configs/
│   ├── keyword-rules.yaml one file per classifier variant
│   └── llm-v1.yaml
├── prompts/
│   └── diet-v1.md         only for LLM classifiers
├── my_rules.py            only for your own Python classifiers
└── runs/                  created by labelbench
```

| File | Required | Purpose |
|---|---|---|
| `task.yaml` | yes | names the columns for id, features and label, and points to `labels.csv` |
| `labels.csv` | yes | one row per allowed label, optionally with a description and coarser levels |
| `gold.csv` | yes | one row per item with id, features and the correct label |
| classifier config | yes | which classifier runs with which parameters |
| classifier code | for your own classifiers | a Python class, see [Writing a classifier](#writing-a-classifier) |
| prompt file | for LLMs | prompt template read by your LLM classifier |

All CSV files are UTF-8 and comma-separated. Text fields may contain commas,
quotes and line breaks as long as they are quoted correctly, which every
spreadsheet program and CSV library does automatically.

## Quickstart

This example classifies canteen meals into four dietary categories.

**1. Describe the task** in `tasks/diet/task.yaml`:

```yaml
name: diet
type: single              # one label per item; "multi" for several
id: id                    # column with a unique id per item
features: [name, notes]   # columns the classifier may see
label: label              # column with the correct label
labels: labels.csv
split: split              # optional, see "Working with dev and test"
```

**2. List the allowed labels** in `tasks/diet/labels.csv`:

```csv
label,description
vegan,"No animal products at all"
vegetarian,"No meat or fish, but may contain dairy or eggs"
meat,"Contains meat"
fish,"Contains fish or seafood, but no meat"
```

**3. Provide the gold standard** in `tasks/diet/gold.csv`:

```csv
id,name,notes,label,split
1,"Spaghetti Bolognese","Rind",meat,dev
2,"Gemüsecurry mit Reis","vegan",vegan,dev
3,"Seelachsfilet mit Kartoffeln","Fisch",fish,test
4,"Käsespätzle","Milch, Ei",vegetarian,dev
```

**4. Choose a classifier** in `configs/keyword-rules.yaml` (the class itself
is shown in the next section):

```yaml
classifier: my_rules:KeywordRules
params:
  meat_words: [rind, schwein, hähnchen, wurst]
  fish_words: [fisch, lachs, seelachs]
```

**5. Run and evaluate:**

```bash
labelbench run --task tasks/diet --clf configs/keyword-rules.yaml
```

labelbench first checks the task (see the next section) and refuses to run
if it finds errors. It then classifies all `dev` items, prints the main
metrics and writes a new run directory such as
`runs/20261009-101500_diet_keyword-rules/`. Open the `report.html` inside it
in any browser.

**6. Change something and compare.** After editing the classifier or creating
a second config, run again and compare both runs:

```bash
labelbench compare runs/20261009-101500_diet_keyword-rules runs/20261009-103200_diet_keyword-rules
```

The comparison lists the items that became correct, those that became wrong,
and whether the difference is larger than chance would explain.

## Checking a task

Gold standards that were built by hand almost always contain inconsistencies,
and some of them distort every evaluation without being visible in the
numbers. `labelbench check` looks for them before any classifier runs:

```bash
labelbench check --task tasks/diet
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

## Writing a classifier

A classifier is a Python class with a `predict` method. It receives a list of
items, each a dictionary with the `id` and the feature columns as strings,
and returns one `Prediction` per item in the same order.

```python
# my_rules.py
from labelbench import Prediction


class KeywordRules:
    def __init__(self, meat_words: list[str], fish_words: list[str]):
        self.meat_words = meat_words
        self.fish_words = fish_words

    def predict(self, items: list[dict]) -> list[Prediction]:
        predictions = []
        for item in items:
            text = f"{item['name']} {item['notes']}".lower()
            if any(word in text for word in self.meat_words):
                label = "meat"
            elif any(word in text for word in self.fish_words):
                label = "fish"
            elif "vegan" in text:
                label = "vegan"
            else:
                label = "vegetarian"
            predictions.append(Prediction(label=label))
        return predictions
```

Everything under `params` in the config is passed to the constructor, so one
class can be evaluated with different parameters without touching the code.
The value of `classifier` is `module:Class`; modules in the directory you run
labelbench from are found automatically.

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
classifier: my_rules:KeywordRules
label_map: {veggie: vegetarian}
params: {...}
```

If the class has a method `prepare(task)`, labelbench calls it once before
`predict`. The argument tells the classifier the task's name, its feature
columns, its levels and the full content of `labels.csv` as `task.labels`,
but never the gold labels. A classifier that needs the list of allowed labels,
such as an LLM classifier that puts them into its prompt, takes it from there.

For large jobs, the config may set `batch_size`; labelbench then calls
`predict` with chunks of that size and reports progress after each one.

## Using an LLM as classifier

An LLM classifier is written like any other classifier. A typical one reads a
prompt template in its constructor, fills in the label list in `prepare` and
sends one request per item in `predict`. Writing the classifier yourself keeps
the choice of provider, model and prompt technique in your hands; a built-in
LLM classifier is planned.

Two practices pay off in every LLM classifier:

- **Restrict the answer to the label list.** Most providers accept a JSON
  schema whose `enum` contains the allowed labels, so the model cannot invent
  a label. Keep the list in `labels.csv` and build the prompt from it, so that
  renaming a label changes prompt, evaluation and report consistently.
- **Cache the responses.** `labelbench.JsonCache` stores each response under
  a hash of model, prompt and parameters. A second run then only sends items
  whose prompt or input changed, and repeating a run costs nothing.

```python
from labelbench import JsonCache, Prediction

cache = JsonCache(".labelbench-cache")
key = JsonCache.key(model=model, prompt=prompt)
response = cache.get(key)
if response is None:
    response = call_the_model(prompt)      # your API call
    cache.put(key, response)
```

Put the model's explanation into `Prediction.raw` and tokens or latency into
`Prediction.meta`; both appear in the report next to each item. If a
parameter of the config is the path of an existing file, such as a prompt
template, labelbench records its checksum in `provenance.json`, so every run
documents exactly which prompt it used.

## Commands

| Command | Purpose |
|---|---|
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
marked as "gold label doubtful" in the item details and exported as CSV, so
the gold standard can be corrected afterwards. The marks are stored in the
reviewer's browser only, which is why the export exists.

The language of the report is set with `language` in `task.yaml`; English
and German are available.

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
