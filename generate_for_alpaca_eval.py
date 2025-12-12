import json
import datasets
from fire import Fire
from functools import partial
from typing import List
from loguru import logger
from path import Path

from utils import (
    generate_together,
    generate_openai,
    generate_with_references,
    generate_response,
    DEBUG, save_checkpoint_json,
)


def process_fn(
    item,
    model,
    reference_models=[],
    temperature=0.7,
    max_tokens=2048,
    rounds=1,
):
# 生成MoA output
    messages = [{"role": "user", "content": item["instruction"]}]

    references = item.get("references", [])

    if len(references) == 0 and len(reference_models) > 0:

        prev_references = []

        for i_round in range(rounds):

            if DEBUG:
                logger.info(
                    f"Round {i_round+1}/{rounds} to collecting reference responses."
                )

            references = []

            for reference_model in reference_models:

                reference = generate_with_references(
                    model=reference_model,
                    messages=messages,
                    references=prev_references,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    generate_fn=generate_response,
                )

                if reference is not None:

                    references.append(reference)

            if i_round < rounds - 1:

                prev_references = references

                references = []

    output = generate_with_references(
        model=model,  # aggregator
        messages=messages, # prompt for aggregator
        references=references, # other reference model's output
        generate_fn=generate_response,
    )
    # 这里移除了model后缀“-together”
    return {"output": output, "generator": model}


def main(
    model: str,
    output_path: str,
    reference_paths: str = None,
    reference_models: str = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    rounds: int = 1,
    num_proc: int = 16,
    batch_size: int = 50,
):

    if reference_paths is None:
        reference_paths = []
    else:
        reference_paths = reference_paths.split(",")

    if reference_models is None:
        reference_models = []
    else:
        reference_models = reference_models.split(",")

    eval_set = (datasets.load_dataset(
        "tatsu-lab/alpaca_eval", "alpaca_eval_gpt4_baseline", trust_remote_code=True
    )["eval"])
    # .select(range(1))
    # 这里只保留前5个样本，方便调试
    eval_set = eval_set.remove_columns(["output", "generator"])

    if len(reference_paths):

        logger.info(f"`reference_paths` provided: {reference_paths}")

        references = []
        for reference_path in reference_paths:
            with open(reference_path) as f:
                reference_responses = json.load(f)
                logger.info(
                    f"Reading reference outputs: {reference_path} ({len(reference_responses)})"
                )
                for i_reference_response, reference_response in enumerate(
                    reference_responses
                ):
                    if len(references) <= i_reference_response:
                        references.append([reference_response["output"]])
                    else:
                        references[i_reference_response].append(
                            reference_response["output"]
                        )

        eval_set = eval_set.add_column(f"references", references)

    elif len(reference_models):

        logger.info(
            f"`reference_models` provided: {reference_models}. Will generate reference responses on-the-fly."
        )

    logger.info(f"Start.")


    all_results = []
    for i in range(0, len(eval_set), batch_size):
        batch_indices = range(i, min(i + batch_size, len(eval_set)))

        try:
            batch = eval_set.select(batch_indices)
            processed_batch = batch.map(
                partial(
                    process_fn,
                    model=model,
                    reference_models=reference_models,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    rounds=rounds,
                ),
                batched=False,
                num_proc=num_proc,
            )
            all_results.extend(processed_batch)

            ckpath = save_checkpoint_json(all_results, Path(output_path)/model/"checkpoints", i//batch_size)
            logger.info(f"Batch {i//batch_size} saved to {ckpath}.")
        except Exception as e:
            logger.error(f"Batch {i//batch_size}, batch_indices: {batch_indices}, failed: {e}")
            print(f"Skipping batch {i // batch_size}, batch_indices: {batch_indices}, continuing...")
            continue


    # eval_set = eval_set.map(
    #     partial(
    #         process_fn,
    #         model=model,
    #         reference_models=reference_models,
    #         temperature=temperature,
    #         max_tokens=max_tokens,
    #         rounds=rounds,
    #     ),
    #     batched=False,
    #     num_proc=num_proc,
    # )

    # logger.info(f"Saving outputs to {output_path}.")
    #
    # try:
    #     eval_set = eval_set.remove_columns(f"references")
    # except Exception as e:
    #     pass
    #
    # with open(output_path, "w") as f:
    #
    #     json.dump(list(eval_set), f, indent=2)


if __name__ == "__main__":

    Fire(main)
