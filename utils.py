import os
import json
import time
import requests
import openai
import copy

import yaml
from loguru import logger
from openai import AzureOpenAI, OpenAI

DEBUG = int(os.environ.get("DEBUG", "1"))


def generate_together(
        model,
        messages,
        max_tokens=2048,
        temperature=0.7,
        streaming=False,
):
    output = None

    for sleep_time in [1, 2, 4, 8, 16, 32]:

        try:

            endpoint = "https://api.together.xyz/v1/chat/completions"

            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}...`) to `{model}`."
                )

            res = requests.post(
                endpoint,
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": (temperature if temperature > 1e-4 else 0),
                    "messages": messages,
                },
                headers={
                    "Authorization": f"Bearer {os.environ.get('TOGETHER_API_KEY')}",
                },
            )
            if "error" in res.json():
                logger.error(res.json())
                if res.json()["error"]["type"] == "invalid_request_error":
                    logger.info("Input + output is longer than max_position_id.")
                    return None

            output = res.json()["choices"][0]["message"]["content"]

            break

        except Exception as e:
            logger.error(e)
            if DEBUG:
                logger.debug(f"Msgs: `{messages}`")

            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    if output is None:
        return output

    output = output.strip()

    if DEBUG:
        logger.debug(f"Output: `{output[:20]}...`.")

    return output


def generate_together_stream(
        model,
        messages,
        max_tokens=2048,
        temperature=0.7,
):
    endpoint = "https://api.together.xyz/v1"
    client = openai.OpenAI(
        api_key=os.environ.get("TOGETHER_API_KEY"), base_url=endpoint
    )
    endpoint = "https://api.together.xyz/v1/chat/completions"
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature if temperature > 1e-4 else 0,
        max_tokens=max_tokens,
        stream=True,  # this time, we set stream=True
    )

    return response


def generate_openai(
        model,
        messages,
        max_tokens=2048,
        temperature=0.7,
):
    client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    for sleep_time in [1, 2, 4, 8, 16, 32]:
        try:

            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}`) to `{model}`."
                )

            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = completion.choices[0].message.content
            break

        except Exception as e:
            logger.error(e)
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    output = output.strip()

    return output


def generate_response(model,
                      messages,
                      max_tokens=2048,
                      temperature=0.7
                      ):
    config = load_config()
    reference_models = config["reference_models"]

    if "gpt" in model.lower():
        client = AzureOpenAI(api_key=reference_models[model]["api_key"],
                             base_url=reference_models[model]["base_url"])  # api_version??
    else:
        client = OpenAI(api_key=reference_models[model]["api_key"], base_url=reference_models[model]["base_url"])

    output = "conditions"
    for sleep_time in [1, 2, 4, 8, 16, 32]:
        try:
            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}`) to `{model}`."
                )
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = response.choices[0].message.content
            # print("Model: ", model, "Response: ", response)
            break
        except Exception as e:
            logger.error(f"生成响应失败: {e}", exc_info=True)
            logger.info(f"Retry in {sleep_time}s..")
            output = f"生成时出错: {str(e)}"
            time.sleep(sleep_time)
        output = output.strip()
    return output


def inject_references_to_messages(
        messages,
        references,
):
    messages = copy.deepcopy(messages)

    system = f"""You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""

    for i, reference in enumerate(references):
        system += f"\n{i + 1}. {reference}"

    if messages[0]["role"] == "system":

        messages[0]["content"] += "\n\n" + system

    else:

        messages = [{"role": "system", "content": system}] + messages

    return messages


def generate_with_references(
        model,
        messages,
        references=[],
        max_tokens=2048,
        temperature=0.7,
        generate_fn=generate_together,
):
    if len(references) > 0:
        messages = inject_references_to_messages(messages, references)

    return generate_fn(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def load_config(path="./config.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing config file: {path}")
    data = yaml.safe_load(open(path))
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping.")
    return data


def save_checkpoint_json(data, checkpoint_dir, batch_num):
    """以JSON格式保存检查点"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{batch_num}.json")
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return checkpoint_path
