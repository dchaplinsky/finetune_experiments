import json
import os
from typing import Any, Dict, List

import evaluate


def read_all_lines(file_name: str) -> List[Dict[str, Any]]:
    all_lines = []
    with open(file_name, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            json_obj = json.loads(line)
            if json_obj["rank"] == 0:
                all_lines.append(json_obj)

    return all_lines


def main(dir_name: str):
    all_files = os.listdir(dir_name)
    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")
    for filename in sorted(all_files):
        if filename.endswith("jsonl"):
            filepath = f"{dir_name}/{filename}"
            print(filepath)
            all_lines_rank_0 = read_all_lines(filepath)
            refs = [[row["ref"]] for row in all_lines_rank_0]
            hyps = [row["hyp"] for row in all_lines_rank_0]
            score_bleu = sacrebleu.compute(predictions=hyps, references=refs)
            score_spbleu_101 = sacrebleu.compute(
                predictions=hyps, references=refs, tokenize="flores101"
            )
            score_spbleu_200 = sacrebleu.compute(
                predictions=hyps, references=refs, tokenize="flores200"
            )
            score_chrf = chrf.compute(predictions=hyps, references=refs)
            score_chrf_word_order_2 = chrf.compute(
                predictions=hyps, references=refs, word_order=2
            )  # word_order = 2
            metrics = [
                {"metric": "bleu"} | score_bleu,
                {"metric": "spbleu-101"} | score_spbleu_101,
                {"metric": "spbleu-200"} | score_spbleu_200,
                {"metric": "chrf"} | score_chrf,
                {"metric": "chrf++"} | score_chrf_word_order_2,
            ]
            with open(
                f"results_for_sales_representative_paniv/{filename.replace('jsonl', 'metrics')}",
                "w",
            ) as file_obj:
                for metric in metrics:
                    file_obj.write(json.dumps(metric) + "\n")


if __name__ == "__main__":
    main("eval-beams-paniv")
