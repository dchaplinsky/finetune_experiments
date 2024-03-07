"""
The main module of the application
"""

import csv
import logging
import time
from typing import Any, Dict, List, Tuple

import ctranslate2  # type: ignore
import evaluate  # type: ignore
import sentencepiece as spm  # type: ignore
import typer  # type: ignore
from typing_extensions import Annotated

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mt-evaluation")
app = typer.Typer()


def log_hr() -> None:
    """
    Log a horizontal line
    """
    logger.debug("*" * 50)


def extract_sentences(
    source_file_path: str,
    validation_field_name: str,
    source_field_name: str,
) -> Tuple[List[str], List[str]]:
    source_sentences: List[str] = []
    validation_sentences: List[str] = []

    with open(source_file_path, "r", encoding="utf-8") as fp_in:
        reader = csv.DictReader(fp_in)
        for row in reader:
            source_sentences.append(row[source_field_name])
            validation_sentences.append(row[validation_field_name])

    logger.debug(f"First source sentence: {source_sentences[0]} \n")
    logger.debug(f"First validation sentence: {validation_sentences[0]}")
    log_hr()

    return source_sentences, validation_sentences


def encode_source_sentences(
    source_sentences: List[str],
    src_prefix: str,
    sp_processor: spm.SentencePieceProcessor,
) -> List[List[str]]:
    source_sentences_tokenized = sp_processor.encode(source_sentences, out_type=str)
    source_sentences_tokenized = [
        [src_prefix] + sentence for sentence in source_sentences_tokenized
    ]
    logger.debug(f"First sentence tokenized: {source_sentences_tokenized[0]}")
    log_hr()

    return source_sentences_tokenized


def translate_source_sentences_multiple_beams(
    source_sentences_tokenized: List[List[str]],
    translation_model_path: str,
    target_prefix: str,
    device: str,
    beam_size: int,
    max_batch_size: int,
    batch_type: str,
) -> List[List[Dict[str, Any]]]:
    translator = ctranslate2.Translator(translation_model_path, device=device)
    target_prefixes = [[target_prefix]] * len(source_sentences_tokenized)
    translations = translator.translate_batch(
        source_sentences_tokenized,
        batch_type=batch_type,
        max_batch_size=max_batch_size,
        beam_size=beam_size,
        target_prefix=target_prefixes,
        return_scores=True,
        num_hypotheses=beam_size,
    )
    translations = [
        [{"tokens": beam["tokens"], "score": beam["score"]} for beam in translation]
        for translation in translations
    ]
    logger.debug(f"First sentence translated: {translations[0]}")
    log_hr()

    return translations


def decode_multiple_beams(
    translated_sentences_tokenized: List[List[Dict[str, Any]]],
    target_prefix: str,
    sp_processor: spm.SentencePieceProcessor,
) -> List[List[str]]:
    """
    Decode the translated sentences using the SentencePieceProcessor
    Args:
        translated_sentences_tokenized: list of translated sentences tokenized
        target_prefix: target prefix
        sp_processor: SentencePieceProcessor
    Returns:
        translations_decoded: list of translated sentences decoded
    """
    translations_decoded = []
    for translation in translated_sentences_tokenized:
        translation_decoded = []
        for beam in translation:
            beam_decoded = {}
            beam_decoded["tokens"] = sp_processor.decode(beam["tokens"])[
                len(target_prefix) + 1 :
            ]
            beam_decoded["score"] = beam["score"]
            translation_decoded.append(beam_decoded)
        translations_decoded.append(translation_decoded)
    log_hr()

    return translations_decoded


def evaluate_bleu_score_per_single_sentence(
    prediction: str, reference: str, bleu: evaluate.EvaluationModule
) -> Dict[str, Any]:
    result = bleu.compute(predictions=[prediction], references=[reference])

    return result


def write_multiple_beams_to_file(
    target_file_path: str,
    source_sentences: List[str],
    validation_sentences: List[str],
    scores: List[List[Dict[str, Any]]],
) -> None:
    total_bleu = 0
    bleu_count = 0
    flatten_scores = []
    for source, reference, translation in zip(
        source_sentences, validation_sentences, scores
    ):
        complete_stats = {}
        complete_stats["src"] = source
        complete_stats["ref"] = reference
        for index, beam in enumerate(translation):
            complete_stats[f"beam_{index}_hyp"] = beam["tokens"]
            complete_stats[f"beam_{index}_logprob"] = beam["score"]
            complete_stats[f"beam_{index}_bleu"] = beam["bleu"]

        max_bleu = max(translation, key=lambda x: x["bleu"])
        complete_stats["max_bleu"] = max_bleu["bleu"]
        total_bleu += max_bleu["bleu"]
        bleu_count += 1
        flatten_scores.append(complete_stats)

    logger.info(f"AVG bleu = {total_bleu/bleu_count}")

    with open(target_file_path, "w", encoding="utf-8") as fp_out:
        writer = csv.DictWriter(fp_out, fieldnames=flatten_scores[0].keys())
        writer.writeheader()
        writer.writerows(flatten_scores)


@app.command()
def eval_model_multpl_beams_ready_prep(
    source_file_path: Annotated[str, typer.Option()],
    preprocessed_file_path: Annotated[str, typer.Option()],
    target_file_path: Annotated[str, typer.Option()],
    translation_model_path: Annotated[str, typer.Option()],
    src_prefix: Annotated[str, typer.Option()],
    target_prefix: Annotated[str, typer.Option()],
    tokenizer_model_path: Annotated[str, typer.Option()],
    validation_field_name: Annotated[str, typer.Option()],
    source_field_name: Annotated[str, typer.Option()],
    device: str = "cpu",
    beam_size: int = 5,
    max_batch_size: int = 200,
    batch_type: str = "tokens",
    target_tokenizer_model_path: Annotated[
        str,
        typer.Option(help="Tokenizer for target language (if different from source)"),
    ] = None,
) -> None:
    start = time.time()
    logger.info(f"Creating SentencePieceProcessor from {tokenizer_model_path}...")

    # Why not tokenizer = Tokenizer.from_file?
    sp_processor = spm.SentencePieceProcessor()
    sp_processor.Load(
        tokenizer_model_path
    )  # I don't know why the linter wants Load not load

    if target_tokenizer_model_path:
        logger.info(
            "Creating SentencePieceProcessor from {target_tokenizer_model_path}...",
        )
        sp_processor_target = spm.SentencePieceProcessor()
        sp_processor_target.Load(target_tokenizer_model_path)
    else:
        sp_processor_target = sp_processor

    logger.info("SentencePieceProcessor was created!")
    log_hr()
    logger.info(
        "Extracting source & validation sentences from file %s...", source_file_path
    )
    reference_sentences, validation_sentences = extract_sentences(
        source_file_path=source_file_path,
        validation_field_name=validation_field_name,
        source_field_name=source_field_name,
    )
    source_sentences = []
    with open(preprocessed_file_path) as f:
        source_sentences = f.readlines()
    logger.info("Sentences extracted!")
    log_hr()
    logger.info("Tokenizing sentences...")
    source_sentences_tokenized = encode_source_sentences(
        source_sentences=source_sentences,
        src_prefix=src_prefix,
        sp_processor=sp_processor,
    )
    logger.info("Tokenization completed!")
    log_hr()
    logger.info("Translating sentences...")
    translated_sentences_tokenized = translate_source_sentences_multiple_beams(
        source_sentences_tokenized=source_sentences_tokenized,
        translation_model_path=translation_model_path,
        target_prefix=target_prefix,
        device=device,
        beam_size=beam_size,
        max_batch_size=max_batch_size,
        batch_type=batch_type,
    )
    logger.info(f"Translation completed in {time.time() - start}")
    log_hr()
    logger.info("Decoding translation...")
    translations_decoded = decode_multiple_beams(
        translated_sentences_tokenized=translated_sentences_tokenized,
        target_prefix=target_prefix,
        sp_processor=sp_processor_target,
    )
    logger.info(f"Decoding completed in {time.time() - start}")
    log_hr()
    logger.info("Evaluating the model...")
    scores = []
    bleu = evaluate.load("bleu")
    for translations, validation in zip(translations_decoded, validation_sentences):
        evaluated_translations = []
        for beam in translations:
            evaluation_result = evaluate_bleu_score_per_single_sentence(
                prediction=beam["tokens"], reference=validation, bleu=bleu
            )
            evaluated_translations.append(beam | {"bleu": evaluation_result["bleu"]})
        scores.append(evaluated_translations)
    logger.info(f"Evaluation completed in {time.time() - start}")
    log_hr()
    write_multiple_beams_to_file(
        target_file_path=target_file_path,
        source_sentences=reference_sentences,
        scores=scores,
        validation_sentences=validation_sentences,
    )
    logger.info("Results saved")
    log_hr()
    logger.info(f"Execution lasted for {time.time() - start}")


if __name__ == "__main__":
    app()
