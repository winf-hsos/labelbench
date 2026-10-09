# @@project@@

An evaluation project for [labelbench](https://github.com/winf-hsos/labelbench),
created with labelbench @@version@@.

## Layout

| Folder | Content | Edit? |
|---|---|---|
| `tasks/` | one folder per task: `task.yaml`, `labels.csv`, `gold.csv` | yes |
| `classifiers/` | Python code of your own classifiers | yes |
| `prompts/` | prompt templates of LLM classifiers | only add new versions |
| `configs/` | one file per classifier variant | only add new versions |
| `runs/` | one folder per run, written by labelbench | no |

A run combines one task with one config. Never change a prompt or config that
was used in a run; copy it to a new version instead, so every run stays
reproducible.

## Commands

```bash
labelbench check --task tasks/example
labelbench run --task tasks/example --clf configs/keywords-v1.yaml
labelbench compare runs/<run_a> runs/<run_b>
labelbench new task NAME
labelbench new classifier NAME --provider anthropic --task tasks/NAME
```

Run all commands from this folder, because classifiers and prompts are found
relative to it.
