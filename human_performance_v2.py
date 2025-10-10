#!/usr/bin/env python3
"""
Replace human_performance.py. Evaluate two human experts and a ViT model on the same subset of images:
– Top-1 / Top-3 accuracy
– Weighted F1
– Cohen’s κ vs. ground truth
– Inter-expert & model-vs-expert κ
"""

import argparse
import sys
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
)

# --- label mapping for human word → numeric L3 codes
REASSIGN_NAME_LABEL_L3 = {
    'Urban': 0,
    'Broadleaved Mixed and Yew Woodland': 1,
    'Coniferous Woodland': 2,
    'Sea': 3,
    'Arable and Horticulture': 4,
    'Improved Grassland': 5,
    'Neutral Grassland': 6,
    'Calcareous Grassland': 7,
    'Acid Grassland': 8,
    'Bracken': 9,
    'Dwarf Shrub Heath': 10,
    'Fen, Marsh, Swamp': 11,
    'Bog': 12,
    'Littoral Rock': 13,
    'Littoral Sediment': 14,
    'Montane': 15,
    'Standing Open Waters and Canals': 16,
    'Inland Rock': 17,
    'Supra-littoral Rock': 18,
    'Supra-littoral Sediment': 19
}


# --- I/O utilities

def load_table(path: str) -> pd.DataFrame:
    """Read CSV or TSV, or exit."""
    for sep in [",", "\t"]:
        try:
            return pd.read_csv(path, sep=sep)
        except Exception:
            continue
    print(f"ERROR: could not read {path}", file=sys.stderr)
    sys.exit(1)


# --- Preparation functions

def prepare_truth(truth_path: str) -> pd.DataFrame:
    """
    Load full ground truth. Expects columns:
      - file_name
      - plot_labels    (numeric)
    Returns DataFrame with columns [file_name, true_label].
    """
    df = load_table(truth_path)
    if "plot_labels" not in df.columns:
        raise KeyError("truth table must contain 'plot_labels'")
    return df[["file_name", "plot_labels"]].rename(
        columns={"plot_labels": "true_label"}
    )


def prepare_human(path: str) -> pd.DataFrame:
    """
    Load an expert’s annotations. Expects columns:
      - Source Name, Top 1 Label, Top 2 Label, Top 3 Label
    Maps word-labels → numeric using REASSIGN_NAME_LABEL_L3.
    Returns columns [file_name, pred1, pred2, pred3]
    """
    df = load_table(path).rename(columns={
        "Source Name": "file_name",
        "Top 1 Label": "raw1",
        "Top 2 Label": "raw2",
        "Top 3 Label": "raw3"
    })
    for i in (1, 2, 3):
        # df[f"pred{i}"] = df[f"raw{i}"].map(REASSIGN_NAME_LABEL_L3)
        df[f"pred{i}"] = df[f"raw{i}"].map(REASSIGN_NAME_LABEL_L3).astype(pd.Int64Dtype())
    return df[["file_name", "pred1", "pred2", "pred3"]]


def prepare_model(ok_path: str, err_path: str) -> pd.DataFrame:
    """
    Load ViT results: two tables (correct vs misclassified).
    Expects columns:
      - file_name
      - top1_label, top2_label, top3_label   (all numeric)
    Returns concatenated [file_name, pred1, pred2, pred3].
    """
    df_ok  = load_table(ok_path)
    df_err = load_table(err_path)
    df = pd.concat([df_ok, df_err], ignore_index=True)
    return df.rename(columns={
        "top3_label_1": "pred1",
        "top3_label_2": "pred2",
        "top3_label_3": "pred3"
    })[["file_name", "pred1", "pred2", "pred3"]]


# --- Merge & evaluation

def merge_with_truth(pred_df: pd.DataFrame,
                     truth_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join predictions to truth; warn+drop orphans.
    Returns DataFrame with [file_name, pred1–3, true_label].
    """
    df = pred_df.merge(truth_df, on="file_name", how="left", indicator=True)
    orphans = df[df["_merge"] == "left_only"]
    if not orphans.empty:
        warnings.warn(
            f"Dropping {len(orphans)} rows missing from truth: "
            # f"{orphans['file_name'].tolist()}"
        )
    return df[df["_merge"] == "both"].drop(columns="_merge")


def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Given merged [pred1–3, true_label], return:
      top1_acc, top3_acc, weighted_f1, kappa_vs_truth
    """
    top1_correct = df["pred1"] == df["true_label"]
    # top3_correct = df.apply(
    #     lambda r: r["true_label"] in (r["pred1"], r["pred2"], r["pred3"]),
    #     axis=1
    # )
    # handle NA in predictions: treat any NA as no match
    match1 = (df["true_label"] == df["pred1"]).fillna(False)
    match2 = (df["true_label"] == df["pred2"]).fillna(False)
    match3 = (df["true_label"] == df["pred3"]).fillna(False)
    top3_correct = match1 | match2 | match3

    return {
        "top1_acc":       top1_correct.mean(),
        "top3_acc":       top3_correct.mean(),
        "weighted_f1":    f1_score(df["true_label"], df["pred1"], average="weighted"),
        "kappa_truth":    cohen_kappa_score(df["true_label"], df["pred1"]),
        "mcc":            matthews_corrcoef(df["true_label"], df["pred1"])
    }


def draw_cm(df: pd.DataFrame, title: str) -> None:
    """
    Row-normalized Top-1 confusion matrix.
    """
    labels = sorted(set(df["true_label"]) | set(df["pred1"]))
    cm = confusion_matrix(df["true_label"], df["pred1"], labels=labels)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, interpolation="nearest")
    ax.set(title=title, xlabel="Predicted", ylabel="True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center")
    fig.colorbar(im, ax=ax, label="Row-normalized")
    plt.tight_layout()
    plt.show()

def plot_cm(df: pd.DataFrame, title: str) -> None:
    """
    Row-normalized Top-1 confusion matrix.
    """

    def _custom_format(x):
        """
        Custom formatting for the confusion matrix, if an entry in the confusion matrix is a float number,
        it shows as with .2f precision.
        :param x:
        :return:
        """
        if x == 0:
            return '0'
        else:
            return f'{x:.2f}'

    labels = sorted(set(df["true_label"]) | set(df["pred1"]))
    # Map numeric labels to their word equivalents
    inv_REASSIGN = {v: k for k, v in REASSIGN_NAME_LABEL_L3.items()}
    word_labels = [inv_REASSIGN.get(lbl, str(lbl)) for lbl in labels]

    cm = confusion_matrix(df["true_label"], df["pred1"], labels=labels)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    annot_data = np.array([[_custom_format(val) for val in row] for row in cm_norm])
    fmt = ''
    # Create plot
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=annot_data, fmt=fmt, cmap='Blues', xticklabels=word_labels, yticklabels=word_labels)
    plt.xticks(rotation=45, ha="right", fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Predicted', fontsize=18)
    plt.ylabel('True', fontsize=18)
    # plt.title(title)
    plt.tight_layout()
    plt.savefig(f"human_vs_ai/{title}.png")
    # plt.show()

def evaluate(name: str,
             pred_df: pd.DataFrame,
             truth_df: pd.DataFrame,
             subset: set = None,
             draw: bool = False) -> pd.DataFrame:
    """
    Merge predictions with truth, restrict to subset if given,
    compute & print metrics, optionally draw CM.
    Returns the merged subset DataFrame.
    """
    df = merge_with_truth(pred_df, truth_df)
    if subset is not None:
        df = df[df["file_name"].isin(subset)]
    if df.empty:
        print(f"{name}: no overlapping samples!", file=sys.stderr)
        return df

    m = compute_metrics(df)
    print(f"\n--- {name} (n={len(df)}) ---")
    print(f"Top-1 accuracy:    {m['top1_acc']:.2%}")
    print(f"Top-3 accuracy:    {m['top3_acc']:.2%}")
    print(f"Weighted F1:       {m['weighted_f1']:.3f}")
    print(f"Cohen’s κ vs truth: {m['kappa_truth']:.3f}")
    print(f"Matthews Correlation: {m['mcc']:.3f}")

    if draw:
        plot_cm(df, title=f"Confusion Matrix: {name}")
    return df


def compute_pairwise_kappa(df1: pd.DataFrame,
                           df2: pd.DataFrame,
                           name1: str,
                           name2: str) -> None:
    """
    Cohen’s κ between pred1 of df1 & df2, aligned by file_name.
    """
    both = df1[["file_name","pred1"]].merge(
        df2[["file_name","pred1"]],
        on="file_name",
        suffixes=(f"_{name1}", f"_{name2}")
    )
    k = cohen_kappa_score(both[f"pred1_{name1}"], both[f"pred1_{name2}"])
    print(f"Cohen’s κ ({name1} vs {name2}): {k:.3f}")

def compute_pairwise_mcc(df1: pd.DataFrame,
                           df2: pd.DataFrame,
                           name1: str,
                           name2: str) -> None:
    """
    MCC between pred1 of df1 & df2, aligned by file_name.
    """
    both = df1[["file_name","pred1"]].merge(
        df2[["file_name","pred1"]],
        on="file_name",
        suffixes=(f"_{name1}", f"_{name2}")
    )
    # print(f"\nMerged predictions (first 5 rows):\n{both.head()}\n")
    # print(f"Columns are: {both.columns.tolist()}\n")

    k = matthews_corrcoef(both[f"pred1_{name1}"], both[f"pred1_{name2}"])
    print(f"MCC ({name1} vs {name2}): {k:.3f}")


# --- Main CLI

def main():
    p = argparse.ArgumentParser(
        description="Evaluate human experts & ViT model on same subset"
    )
    p.add_argument("--expert1", required=True, help="Expert 1 CSV/TSV")
    p.add_argument("--expert2", required=True, help="Expert 2 CSV/TSV")
    p.add_argument("--model_ok", required=True,
                   help="ViT correct-predictions CSV/TSV")
    p.add_argument("--model_err", required=True,
                   help="ViT misclassified CSV/TSV")
    p.add_argument("--truth", required=True, help="Full truth CSV/TSV")

    args = p.parse_args()

    truth_df = prepare_truth(args.truth)
    human1   = prepare_human(args.expert1)
    human2   = prepare_human(args.expert2)
    # print(human1)

    # # Identify any missing pred1 mappings for Expert1
    # missing_pred1 = human2[human2['pred1'].isna()]
    # if not missing_pred1.empty:
    #     print("Expert 2: pred1 NA for files:", missing_pred1['file_name'].tolist())
    model    = prepare_model(args.model_ok, args.model_err)

    # 1) Evaluate experts
    df1 = evaluate("Expert 1", human1, truth_df)
    df2 = evaluate("Expert 2", human2, truth_df)

    # 2) Inter-expert κ and mcc
    compute_pairwise_kappa(df1, df2, "Expert1", "Expert2")
    compute_pairwise_mcc(df1, df2, "Expert1", "Expert2")

    # 3) Evaluate model on same subset
    subset = set(df1["file_name"])
    dfm = evaluate("Vit model", model, truth_df, subset=subset)

    # 4) Model vs. each expert κ and mcc
    compute_pairwise_kappa(dfm, df1, "ViT", "Expert1")
    compute_pairwise_kappa(dfm, df2, "ViT", "Expert2")

    compute_pairwise_mcc(dfm, df1, "ViT", "Expert1")
    compute_pairwise_mcc(dfm, df2, "ViT", "Expert2")


if __name__ == "__main__":
    main()
