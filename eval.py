#
# Milestone 4: Evaluation
# 
# John Kendall
# 
# Due: March 29, 2026

from argparse import ArgumentParser, Namespace
from collections import defaultdict
from pathlib import Path

from utils import load_gold_keys, load_predictions, calculate_accuracy

ACC_CATEGORIES = ("Acc", "Noun", "Verb")


def make_parser() -> Namespace:
    parser = ArgumentParser(
        description='Evaluate the predictions of a WSD model using gold senses for each instance.'
    )
    parser.add_argument(
        "GOLD_KEY_FILE",
        type=Path,
        help="File with each line consistenting of space-separated `instanceID`, and one or more gold sense keys."
    )
    parser.add_argument(
        "PRED_FILES",
        type=Path,
        nargs="+",
        help="Files of predicted senses. Each line is `instanceID pos predicted_sense`."
    )

    return parser.parse_args()


def main(pred_paths: list[Path], gold_path: Path) -> None:

    GOLD = load_gold_keys(gold_path)

    results = []

    all_metrics = {}
    
    for file in pred_paths:

        PREDS = load_predictions(file)

        totals = build_results_dict(PREDS, GOLD)

        all_metrics[file.stem] = totals

        outstring = file.stem
        for pos in sorted(totals):
            pos_acc = calculate_accuracy(totals[pos])
            outstring += f"\n{pos.title()}: {pos_acc:.02f}"
        
        results.append(outstring)
    
    print(*results, sep='\n\n')

    generate_latex_table(all_metrics)


def build_results_dict(
        preds: list[tuple[str, str, str]],
        gold_keys: dict[str, set[str]],
) -> dict[str, dict[bool, int]]:
    
    totals = defaultdict(lambda: {True: 0, False: 0})
    
    for iid, pos, pred_sense in preds:
        gold_senses = gold_keys.get(iid)
        is_correct = pred_sense in gold_senses
    
        totals["acc"][is_correct] += 1
        totals[pos.lower()][is_correct] += 1
    
    return totals


def generate_latex_table(
        all_metrics: dict[str, dict[str, str]],
        metric_cols: list[str] = ACC_CATEGORIES,
        prec: int = 2,
) -> None:
    header = ["Model"] + list(metric_cols)

    rows = [header]
    for model, pos_counts in all_metrics.items():
        row = [model.replace("_", "\\_")]
        
        for col in metric_cols:
            counts = pos_counts.get(col.lower())

            if not counts:
                row.append('-')
                continue
            
            row.append(f"{(calculate_accuracy(counts) * 100):.0{prec}f}")
        
        rows.append(row)

    print("\n--- COPY THIS INTO .TEX FILE ---\n")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{|l|" + "c|" * len(metric_cols) + "}")
    print("\\hline")

    row_end = " \\\\ \\hline"
    col_sep = " & "

    table = '\n'.join(col_sep.join(row) + row_end for row in rows)

    print(table)
        
    print("\\end{tabular}")
    print("\\caption{WSD Accuracy (Overall and by Part-of-Speech)}")
    print("\\label{tab:results}")
    print("\\end{table}")
    print("\n--------------------------------------\n")



if __name__ == "__main__":
    opts = make_parser()
    GOLD_PATH: Path = opts.GOLD_KEY_FILE
    PRED_PATH: list[Path] = opts.PRED_FILES

    main(PRED_PATH, GOLD_PATH)