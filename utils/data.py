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


def load_data_with_keyword(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (train_df, dev_df) each with columns: text, binary_label (0/1), keyword.
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

    train_df = df.loc[df.index.isin(train_ids), ["text", "binary_label", "keyword"]].copy()
    dev_df = df.loc[df.index.isin(dev_ids), ["text", "binary_label", "keyword"]].copy()

    return train_df, dev_df


# Canonical ordering of PCL categories (alphabetical for reproducibility).
# All 7 categories present in dontpatronizeme_categories.tsv.
PCL_CATEGORIES = [
    "Authority_voice",
    "Compassion",
    "Metaphors",
    "Presupposition",
    "Shallow_solution",
    "The_poorer_the_merrier",
    "Unbalanced_power_relations",
]


def load_data_categories(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (train_df, dev_df) with columns:
      text, binary_label (0/1), and one binary column per PCL category.

    Category columns are named by PCL_CATEGORIES (e.g. 'Authority_voice').
    A paragraph gets 1 in a category column if any annotated span in that
    paragraph belongs to that category; 0 otherwise.
    Paragraphs with no category annotations (non-PCL) get 0 in all columns.

    Used for Exp C: multi-task PCL binary + per-category classification.
    """
    # Load main PCL data (text + binary_label)
    col_names = ["par_id", "art_id", "keyword", "country_code", "text", "label"]
    df = pd.read_csv(
        os.path.join(data_dir, "dontpatronizeme_pcl.tsv"),
        sep="\t", skiprows=4, names=col_names, index_col="par_id"
    )
    df.dropna(inplace=True)
    df["binary_label"] = (df["label"] >= 2).astype(int)
    df["text"] = df["text"].apply(clean_text)

    # Load categories file and aggregate to paragraph level
    cat_col_names = [
        "par_id", "art_id", "text_cat", "keyword", "country_code",
        "span_start", "span_finish", "span_text", "pcl_category", "num_annotators",
    ]
    cat_df = pd.read_csv(
        os.path.join(data_dir, "dontpatronizeme_categories.tsv"),
        sep="\t", skiprows=4, names=cat_col_names,
    )
    # Build multi-hot category matrix: one row per par_id, one col per category
    cat_df["value"] = 1
    cat_pivot = (
        cat_df.pivot_table(
            index="par_id", columns="pcl_category", values="value",
            aggfunc="max", fill_value=0,
        )
        .reindex(columns=PCL_CATEGORIES, fill_value=0)
    )

    # Left-join: all paragraphs get category columns; non-annotated get 0
    df = df.join(cat_pivot, how="left")
    for col in PCL_CATEGORIES:
        df[col] = df[col].fillna(0).astype(int)

    train_ids = pd.read_csv(os.path.join(data_dir, "train_semeval_parids-labels.csv"))["par_id"].values
    dev_ids = pd.read_csv(os.path.join(data_dir, "dev_semeval_parids-labels.csv"))["par_id"].values

    keep_cols = ["text", "binary_label"] + PCL_CATEGORIES
    train_df = df.loc[df.index.isin(train_ids), keep_cols].copy()
    dev_df   = df.loc[df.index.isin(dev_ids),   keep_cols].copy()

    return train_df, dev_df
