import os
import re
import pandas as pd
from html import unescape

def clean_text(text: str) -> str:
    """Remove HTML noise, such as <h>/</h> tags, @@digits artifacts, and collapse whitespace."""
    text = unescape(text)
    text = re.sub(r"</?h>", "", text)
    text = re.sub(r"@@\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (train_df, dev_df) each with columns: text, binary_label (0/1).
    Index is par_id.
    """
    col_names = ["par_id", "art_id", "keyword", "country_code", "text", "label"]
    df = pd.read_csv(
        os.path.join(data_dir, "dontpatronizeme_pcl.tsv"),
        sep="\t", skiprows=4, names=col_names, index_col="par_id"
    )
    df.dropna(inplace=True)
    df["binary_label"] = (df["label"] >= 2).astype(int)
    df["text"] = df["text"].apply(clean_text)

    train_ids = pd.read_csv(os.path.join(data_dir, "train_semeval_parids-labels.csv"))["par_id"].values
    dev_ids = pd.read_csv(os.path.join(data_dir, "dev_semeval_parids-labels.csv"))["par_id"].values

    train_df = df.loc[df.index.isin(train_ids), ["text", "binary_label"]].copy()
    dev_df = df.loc[df.index.isin(dev_ids), ["text", "binary_label"]].copy()

    return train_df, dev_df
