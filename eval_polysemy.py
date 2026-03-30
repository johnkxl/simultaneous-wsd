#
# Milestone 4: Additional Evaluation (Polysemy)
#
# John Kendall
# March 29, 2026

import argparse
from pathlib import Path

from utils import load_dataset, get_synsets, load_gold_keys, load_predictions, calculate_accuracy


def make_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate WSD predictions by WordNet polysemy bins.'
    )
    parser.add_argument(
        "DATASET_XML",
        type=Path,
        help="Path to the original dataset XML file (e.g., semeval2007.data.xml) to extract lemmas."
    )
    parser.add_argument(
        "GOLD_KEY_FILE",
        type=Path,
        help="File with each line consisting of space-separated `instanceID`, and one or more gold sense keys."
    )
    parser.add_argument(
        "PRED_FILES",
        type=Path,
        nargs="+",
        help="Files of predicted senses. Each line is `instanceID pos predicted_sense`."
    )
    return parser.parse_args()


def build_polysemy_map(dataset_xml: Path) -> dict[str, str]:
    """
    Parses the dataset and maps each target ID to its polysemy bin.
    """
    targets, _ = load_dataset(dataset_xml)
    poly_map = {}
    
    for target in targets:
        synsets = get_synsets(target.word.lemma, target.word.pos)
        count = len(synsets)
        
        if count <= 3:
            poly_map[target.id] = "Low"
        elif count <= 8:
            poly_map[target.id] = "Medium"
        else:
            poly_map[target.id] = "High"
            
    return poly_map


def build_polysemy_results_dict(
        preds: list[tuple[str, str, str]],
        gold_keys: dict[str, set[str]],
        poly_map: dict[str, str]
) -> dict[str, dict[bool, int]]:
    """
    Calculates correct/incorrect predictions binned by polysemy.
    """
    totals = {
        "Low": {True: 0, False: 0},
        "Medium": {True: 0, False: 0},
        "High": {True: 0, False: 0}
    }
    
    for iid, _, pred_sense in preds:
        if iid not in gold_keys or iid not in poly_map:
            continue
            
        gold_senses = gold_keys.get(iid)
        is_correct = pred_sense in gold_senses
        
        bin_name = poly_map[iid]
        totals[bin_name][is_correct] += 1
        
    return totals


def generate_polysemy_latex_table(
        all_metrics: dict[str, dict[str, dict[bool, int]]],
        prec: int = 2
) -> None:
    
    bins = ["Low", "Medium", "High"]
    headers = ["Model", "Low (1-3)", "Medium (4-8)", "High ($\\geq$ 9)"]

    rows = [headers]
    for model, bin_counts in all_metrics.items():
        row = [model.replace("_", "\\_")]
        
        for b in bins:
            counts = bin_counts.get(b)
            if sum(counts.values()) == 0:
                row.append('-')
            else:
                acc = calculate_accuracy(counts) * 100
                row.append(f"{acc:.0{prec}f}")
        
        rows.append(row)

    print("\n--- COPY THIS INTO .TEX FILE ---\n")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{|l|c|c|c|}")
    print("\\hline")

    row_end = " \\\\ \\hline"
    col_sep = " & "

    table = '\n'.join(col_sep.join(row) + row_end for row in rows)

    print(table)
        
    print("\\end{tabular}")
    print("\\caption{WSD Accuracy Grouped by Target Word Polysemy}")
    print("\\label{tab:polysemy}")
    print("\\end{table}")
    print("\n--------------------------------------\n")


def main(dataset_path: Path, gold_path: Path, pred_paths: list[Path]) -> None:
    print("Building polysemy map from dataset...")
    poly_map = build_polysemy_map(dataset_path)
    
    gold_keys = load_gold_keys(gold_path)
    all_metrics = {}
    
    for file in pred_paths:
        preds = load_predictions(file)
        totals = build_polysemy_results_dict(preds, gold_keys, poly_map)
        all_metrics[file.stem] = totals
        
        print(f"\n--- {file.stem} ---")
        for b in ["Low", "Medium", "High"]:
            if sum(totals[b].values()) > 0:
                print(f"{b}: {calculate_accuracy(totals[b])*100:.2f}%")
            else:
                print(f"{b}: N/A")

    generate_polysemy_latex_table(all_metrics)


if __name__ == "__main__":
    opts = make_parser()
    main(opts.DATASET_XML, opts.GOLD_KEY_FILE, opts.PRED_FILES)