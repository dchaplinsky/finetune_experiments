import csv
import json
import time
import argparse
from typing import Any, Dict, List
from itertools import islice
from pathlib import Path

import evaluate
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.generation import BeamSearchDecoderOnlyOutput


SYSTEM_PROMPT_TEMPLATE = """
You are professional English to Ukrainian translator, complete the translation according domain examples
###

{translation_few_shot}

###

English: {original}
Translation:
"""

sacrebleu = evaluate.load("sacrebleu")


def register(parser):
    parser.add_argument("--base_model", default="mistralai/Mistral-7B-v0.1", type=str)
    parser.add_argument("--output_dir", default="results", type=Path)
    parser.add_argument("--subset", default="devtest", type=str)
    parser.add_argument(
        "--batch_size",
        default=2,
        type=int,
        help="should be small enough to accommodate all beams",
    )
    parser.add_argument("--beams", default=10, type=int)

    return parser


def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


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


def prepare_prompt(query_result: Dict[str, Any], references_limit: int):
    original = query_result["orig"]
    context_pairs: List[Dict[str, str]] = query_result["context"]
    translation_few_shot = ""
    for translation_pair in context_pairs[:references_limit]:
        translation_few_shot += f"English: {translation_pair['orig']}\n"
        translation_few_shot += f"Translation: {translation_pair['trans']}\n\n"

    return SYSTEM_PROMPT_TEMPLATE.format(
        translation_few_shot=translation_few_shot, original=original
    )


def make_result():
    return {
        "id": [],
        "rank": [],
        "logprob": [],
        "src": [],
        "hyp": [],
        "ref": [],
        "bleu": [],
    }


def main(
    flores_context_path: str = "data/flores_context/context_floresdev_sbert_loose.jsonl",
    references_limit: int = 10,
    beams: int = 2,
    model_max_length: int = 1024,
):
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1", device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        model_max_length=1024,
        use_fast=False,
        add_eos_token=False,
        add_bos_token=False,
        pad_token="<s>",
        padding_side="left",
    )
    start_time = time.time()
    all_query_results = read_all_lines(flores_context_path)
    all_query_results = all_query_results[:10]

    all_prompts = []
    sources = []
    references = []
    all_token_counts = []
    ids = list(range(1, len(all_query_results) + 1))
    for query_result in tqdm(all_query_results):
        temp_references_limit = references_limit
        translation_prompt = prepare_prompt(query_result, references_limit)
        sources.append(query_result["orig"])
        references.append(query_result["trans"])
        model_input = tokenizer([translation_prompt], return_tensors="pt").to("cpu")
        tokens_count = len(model_input.input_ids[0])
        while tokens_count > model_max_length or temp_references_limit == 0:
            temp_references_limit -= 1
            translation_prompt = prepare_prompt(query_result, temp_references_limit)
            model_input = tokenizer([translation_prompt], return_tensors="pt").to("cpu")
            tokens_count = len(model_input.input_ids[0])
        all_token_counts.append(tokens_count)
        all_prompts.append(translation_prompt)
    print(f"Max tokens = {max(all_token_counts)}")
    inputs = tokenizer(all_prompts, return_tensors="pt", padding=True)
    model.to("cuda")
    outputs: BeamSearchDecoderOnlyOutput = model.generate(
        input_ids=inputs["input_ids"].cuda(),
        # input_ids=inputs["input_ids"],
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=256,
        use_cache=True,
        generation_config=GenerationConfig(
            pad_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=beams,
            num_return_sequences=beams,
        ),
    )
    logprobs = model.compute_transition_scores(
        outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=True
    ).sum(dim=-1)
    strings = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    result = make_result()
    for example_id, src, ref, batch in zip(
        ids, sources, references, batched(zip(logprobs, strings), beams)
    ):
        for rank, (logprob, output) in enumerate(batch):
            if "[/INST]" in output:
                _, output = output.split("[/INST]", 1)
                output = output.strip()
            else:
                output = "##ERROR"
            print(example_id, logprob.item(), output)
            result["id"].append(example_id)
            result["rank"].append(rank)
            result["logprob"].append(logprob.item())
            result["src"].append(src)
            result["ref"].append(ref)
            result["hyp"].append(output)
            result["bleu"].append(
                sacrebleu.compute(predictions=[output], references=[ref])["score"]
            )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = register(parser)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        args.output_dir / f"beam{args.beams}.{args.subset}.jsonl"
    )
    dataset = main()
    # measure top-1 bleu
    dataset = dataset.filter(lambda x: x["rank"] == 0, load_from_cache_file=False)
    results = sacrebleu.compute(predictions=dataset["hyp"], references=dataset["ref"])
    output_path.with_suffix(".results").write_text(
        json.dumps(results, ensure_ascii=False)
    )
    print(results)
