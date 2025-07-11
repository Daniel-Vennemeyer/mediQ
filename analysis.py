import json
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from collections import defaultdict
from lifelines import LogNormalAFTFitter

def load_data(json_path):
    data = []
    with open(json_path, 'r') as f:
        for line in f:
            if line.strip():  # skip empty lines
                data.append(json.loads(line))
    return data

def compute_cumulative_ig(temp_info_list, upto=None):
    cumulative_ig = defaultdict(float)
    for i, turn in enumerate(temp_info_list):
        if upto is not None and i >= upto:
            break
        ig = turn.get("ig_metrics", {})
        for key, val in ig.items():
            cumulative_ig[key] += val
    return dict(cumulative_ig)

def extract_metrics(entry):
    temp_info = entry.get("interactive_system", {}).get("temp_additional_info", [])
    correct_idx = entry.get("info", {}).get("correct_answer_idx")
    intermediate_choices = entry.get("interactive_system", {}).get("intermediate_choices", [])
    first_correct_turn = next((i for i, choice in enumerate(intermediate_choices) if choice == correct_idx), -1)

    # Ignore examples where the model was right from the start
    if first_correct_turn == 0:
        return []

    records = []
    for i in range(len(temp_info)):
        cumulative_ig = compute_cumulative_ig(temp_info, upto=i + 1)
        ig = cumulative_ig.copy()
        ig.pop("correct", None)  # Remove any inherited 'correct' key from ig_metrics

        correct = 1 if first_correct_turn >= 0 and i >= first_correct_turn else 0

        records.append({
            "turn": i,
            "correct": correct,
            "first_correct_turn": first_correct_turn,
            **ig
        })
    return records

def main(json_path):
    data = load_data(json_path)
    records = [r for entry in data for r in extract_metrics(entry)]

    df = pd.DataFrame(records).fillna(0)

    # Hybrid score computation and correlation
    if all(col in df.columns for col in [
        "kl", "wasserstein", "tv", "cosine",
        "euclidean", "entropy_after"
    ]):
        df["hybrid_score"] = (
            +1.4377 * df["tv"] +
            +0.6922 * df["gini_drop"] +
            -1.1178 * df["margin_gain"]
            # +0.2 * df["cosine"] +
            # +0.3 * df["euclidean"] +
            # +0.5 * df["entropy_after"]
        )
        spearman_corr, spearman_p = spearmanr(df["hybrid_score"], df["correct"])
        print(f"\nhybrid_score              Spearman ρ = {spearman_corr:.3f} (p={spearman_p:.3g})")

    print("\n=== Correlation Between IG Metrics and Final Accuracy ===\n")
    for metric in df.columns:
        if metric == "correct":
            continue
        if df[metric].nunique() <= 1:
            print(f"{metric:25} Constant column (no variance) — correlation not defined.")
            continue
        if df['correct'].nunique() <= 1:
            print(f"{metric:25} Correlation not computed — 'correct' has no variance.")
            continue
        spearman_corr, spearman_p = spearmanr(df[metric], df["correct"])
        pearson_corr, pearson_p = pearsonr(df[metric], df["correct"])
        # print(f"{metric:25} Spearman ρ = {spearman_corr:.3f} (p={spearman_p:.3g}), "
            #   f"Pearson r = {pearson_corr:.3f} (p={pearson_p:.3g})")
        print(f"{metric:25} Spearman ρ = {spearman_corr:.3f} (p={spearman_p:.3g})")

    print("\n=== Log-Normal AFT Regression (Cumulative IG → First Correct Turn) ===\n")

    # Filter to examples that have nonzero first_correct_turn
    aft_df = df[df["first_correct_turn"] > 0].copy()

    if aft_df.empty:
        print("No data with nonzero first_correct_turn for AFT regression.")
    else:
        aft_fitter = LogNormalAFTFitter()
        features = [col for col in aft_df.columns if col not in {"turn", "correct", "first_correct_turn"}]

        # Drop highly correlated features (|correlation| > 0.9)
        corr_matrix = aft_df[features].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
        print(f"Dropping highly correlated features: {to_drop}")
        features = [f for f in features if f not in to_drop]

        aft_fitter.fit(aft_df[features + ["first_correct_turn"]], duration_col="first_correct_turn")

        # Display the z-statistics for interpretability
        summary = aft_fitter.summary
        for metric in features:
            matching_rows = summary.loc[[metric in str(idx) for idx in summary.index]]
            if not matching_rows.empty:
                z = matching_rows.iloc[0]["z"]
                p = matching_rows.iloc[0]["p"]
                print(f"{metric:25} z = {z:.3f}, p = {p:.3g}")
                # Interpret effect size as percent change in expected game length per 1 std increase
                stds = aft_df[metric].std()
                coef = matching_rows.iloc[0]["coef"]
                pct_change = (np.exp(coef * stds) - 1) * 100
                print(f"{metric:25} : 1 stdv increase → {pct_change:.1f}% change in expected game length")

    output_csv = os.path.splitext(json_path)[0] + "_ig_analysis.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")

main("output/output.jsonl")