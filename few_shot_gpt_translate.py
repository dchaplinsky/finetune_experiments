import csv
import json
import time
from typing import Any, Dict, List

import evaluate
from openai import OpenAI
from tqdm import tqdm

SYSTEM_PROMPT_TEMPLATE = """
You are professional English to Ukrainian translator, complete the translation according domain examples
###

{translation_few_shot}

###

English: {original}
Translation:
"""


def write_to_file(
    target_file_path: str,
    source_sentences: List[str],
    translation_sentences: List[str],
    validation_sentences: List[str],
    bleu_scores: List[float],
) -> None:
    """
    Write the evaluation results to a file
    Args:
        target_file_path: path to the target file
        source_sentences: list of source sentences
        translation_sentences: list of translated sentences
        validation_sentences: list of validation sentences
        metrics_evaluated: dictionary of metrics evaluated
    Returns:
        None
    """
    evaluation_entity: Dict[str, List[Any]] = {
        "source": source_sentences,
        "original_translation": validation_sentences,
        "mt_translation": translation_sentences,
        "bleu": bleu_scores,
    }

    evaluation_entity_list: List[Dict[str, Any]] = [
        dict(zip(evaluation_entity, t)) for t in zip(*evaluation_entity.values())
    ]

    with open(target_file_path, "w", encoding="utf-8") as fp_out:
        writer = csv.DictWriter(fp_out, fieldnames=evaluation_entity_list[0].keys())
        writer.writeheader()
        writer.writerows(evaluation_entity_list)


def read_all_lines(file_name: str) -> List[Dict[str, Any]]:
    all_lines = []
    with open(file_name, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            json_obj = json.loads(line)
            all_lines.append(json_obj)

    return all_lines


def prepare_prompt(query_result: Dict[str, Any]):
    original = query_result["orig"]
    context_pairs: List[Dict[str, str]] = query_result["context"]
    translation_few_shot = ""
    for translation_pair in context_pairs:
        translation_few_shot += f"English: {translation_pair['orig']}\n"
        translation_few_shot += f"Translation: {translation_pair['trans']}\n\n"

    return SYSTEM_PROMPT_TEMPLATE.format(
        translation_few_shot=translation_few_shot, original=original
    )


def main():
    all_scores = []
    start_time = time.time()
    sacrebleu = evaluate.load("sacrebleu")
    client = OpenAI(
        api_key="*****",
    )
    all_query_results = read_all_lines(
        "data/flores_context/context_floresdev_sbert_loose.jsonl"
    )
    for query_result in tqdm(all_query_results):
        translation_prompt = prepare_prompt(query_result)
        completion = client.chat.completions.create(
            model="gpt-4",
            # model="gpt-4-turbo-preview",
            # model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": translation_prompt}],
        )
        score = {}
        score["src"] = query_result["orig"]
        score["ref"] = query_result["trans"]
        score["hyp"] = completion.choices[0].message.content
        score["sacrebleu"] = sacrebleu.compute(
            predictions=[score["hyp"]], references=[score["ref"]]
        )["score"]
        all_scores.append(score)

    references = [score["ref"] for score in all_scores]
    translations = [score["hyp"] for score in all_scores]
    source_sentences = [score["src"] for score in all_scores]
    translation_sentences = [score["hyp"] for score in all_scores]
    validation_sentences = [score["ref"] for score in all_scores]
    sacrebleu_scores = [score["sacrebleu"] for score in all_scores]
    write_to_file(
        target_file_path="results/context_floresdev_sbert_loose_scores.csv",
        source_sentences=source_sentences,
        translation_sentences=translation_sentences,
        validation_sentences=validation_sentences,
        bleu_scores=sacrebleu_scores,
    )
    evaluation_result_sacrebleu = sacrebleu.compute(
        predictions=translations, references=references
    )
    print(evaluation_result_sacrebleu)
    print(f"Execution lasted for {time.time() - start_time}")


main()
