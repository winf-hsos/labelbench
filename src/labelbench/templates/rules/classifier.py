"""@@name@@: a rule-based classifier created by `labelbench new classifier`.

It assigns the first label whose keywords occur in an item's text, in the
order of the `keywords` mapping in the config file. Adjust the keywords
there, or replace the logic in predict() with your own rules: labelbench only
needs predict() to return one Prediction per item, in the same order.
"""

from labelbench import Prediction, TaskInfo


class KeywordClassifier:
    def __init__(self, keywords: dict[str, list[str]], fallback: str | None = None,
                 fields: list[str] | None = None) -> None:
        self.keywords = {label: [w.lower() for w in words] for label, words in keywords.items()}
        self.fallback = fallback
        self.fields = fields

    def prepare(self, task: TaskInfo) -> None:
        """Called once before predict(); `task.labels` is the content of labels.csv."""
        known = {row["label"] for row in task.labels}
        unknown = [label for label in self.keywords if label not in known]
        if unknown:
            raise ValueError(f"Keywords given for labels that are not in labels.csv: {unknown}")
        if self.fields is None:
            self.fields = task.features

    def predict(self, items: list[dict[str, str]]) -> list[Prediction]:
        predictions = []
        for item in items:
            text = " ".join(item.get(f, "") for f in self.fields).lower()
            hits = [label for label, words in self.keywords.items()
                    if any(word in text for word in words)]
            label = hits[0] if hits else self.fallback
            predictions.append(Prediction(
                label=label,
                candidates=hits or None,
                raw="matched: " + (", ".join(hits) if hits else "nothing"),
            ))
        return predictions
