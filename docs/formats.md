# File formats

These formats are the contract between tasks, classifiers and the evaluator.
The evaluator and the report read only these files, so any classifier that
writes a valid `predictions.csv` can be evaluated, whatever language it is
written in.

All CSV files are UTF-8, comma-separated and quoted according to RFC 4180.
Text fields may contain commas, quotes and line breaks, so they must always be
read with a real CSV parser (Python's `csv` module, pandas, `readr`) and never
with line-based tools such as `wc -l`, `cut` or `awk`.

Fields that hold structured values (label lists, metadata) contain JSON.

## Task directory

A task is a directory with three files. Tasks live in the project that uses
labelbench, not in this repository.

```
my-task/
  task.yaml
  labels.csv
  gold.csv
```

### task.yaml

```yaml
name: standard-dish            # used in run directory names
description: Map canteen meals to one of ~200 standard dishes.
type: single                   # single | multi

id: unit_id                    # column in gold.csv, unique
features: [name, category, notes]   # columns passed to the classifier
label: label                   # column holding the gold label
labels: labels.csv             # label schema, see below
levels: [group, diet]          # optional: coarser levels, columns in labels.csv

level_gold: {diet: diet_item}  # optional: per-item gold value for a level

gold: gold.csv                 # optional, default gold.csv
weight: n_entries              # optional: frequency weight per row
split: split                   # optional: column with dev / test
show: [comment]                # optional: shown in the report, never given to classifiers
language: de                   # optional: report language, en (default) or de
```

Only `name`, `type`, `id`, `features`, `label` and `labels` are required.
`type: multi` and an `annotators` key for several independent labels per item
are planned but not implemented yet.

### labels.csv

One row per allowed label. The `label` column holds the identifier that
appears in `gold.csv` and `predictions.csv`. Every column named in `levels`
maps the label to a coarser one, so a hierarchy is just additional columns.

```csv
label,description,group,diet
spaghetti_bolognese,"Spaghetti with minced meat sauce",pasta_meat,meat
penne_arrabbiata,"Penne in spicy tomato sauce",pasta_veg,vegan
no_standard_dish,"Not assignable to any standard dish",none,unknown
```

`description` is optional but recommended: it is shown in the report and an
LLM classifier can use it to build its prompt, so the label schema stays the
single source of truth.

### gold.csv

One row per item. It contains the `id` column, all feature columns, the label
column and every optional column named in `task.yaml`.

For the planned `type: multi` the label column will hold a JSON array, e.g.
`["gluten","milk"]`, where an empty array `[]` means "no label applies".

## Items passed to a classifier

The runner hands each classifier a list of items, each a mapping with the
`id` and the feature columns as strings. Gold labels, weights and splits are
never passed to the classifier.

For classifiers in other languages the runner will write the same data as
`items.csv` (columns `id` plus features) and expect `predictions.csv` back;
this adapter is planned.

## predictions.csv

| column       | required | content |
|--------------|----------|---------|
| `id`         | yes      | item id from `gold.csv` |
| `label`      | yes      | predicted label; JSON array for `type: multi`; empty if the classifier gave no valid answer |
| `candidates` | no       | JSON array of labels in ranked order, for top-k metrics |
| `raw`        | no       | raw classifier output, e.g. the LLM response including its reasoning |
| `meta`       | no       | JSON object, e.g. `{"input_tokens": 812, "latency_ms": 940}` |

A prediction that is not part of the label schema is not dropped: it counts
as the reserved class `__invalid__`, so a classifier cannot look better by
answering nonsense on hard cases.

Every item of the evaluated split must appear exactly once.

## Run directory

Each run writes a new directory and never touches an existing one.

```
runs/20261006-143012_standard-dish_llm-v3/
  task.yaml          # copy of the task, pointing to the two copies below
  labels.csv         # copy of the label schema
  gold.csv           # the evaluated gold rows only
  checks.json        # findings of the consistency checks on the full task
  config.yaml        # the classifier configuration as used
  provenance.json    # checksums of task files, config and prompt files; split, runtime
  predictions.csv
  metrics.json
  report.html        # self-contained, no server needed
```

Because the directory contains everything the evaluation needs,
`labelbench report` and `labelbench compare` read nothing else.

### checks.json

A list of findings, each with `severity` (`error`, `warning`, `info`), `code`,
`message`, `count` and up to 25 `examples`. The codes are listed in the
README.

Evaluations of the `test` split are additionally appended to
`runs/test_access.log`, so it can be shown afterwards how often the held-out
data was looked at.
