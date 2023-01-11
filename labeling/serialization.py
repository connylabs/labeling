from pathlib import Path
import pandas as pd


def read_jsonl(filepath):
    filepath = Path(filepath)
    if not filepath.exists():
        return {}
    label_df = pd.read_json(filepath, lines=True).set_index("file_name")
    return label_df.to_dict("index")


def write_jsonl(img_dir, labels):
    img_dir = Path(img_dir)
    labels_path = img_dir.joinpath("metadata.jsonl")

    if not isinstance(labels, dict):
        raise ValueError(f"Expected `labels` to be dict but found {labels}")

    labels = pd.DataFrame.from_dict(labels, orient="index")
    labels = labels.reset_index().rename(columns={"index": "file_name"})
    labels.to_json(labels_path, lines=True, orient="records")
    return labels_path
