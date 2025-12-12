# Advanced Mixture-of-Agents example – 3 layers
# import asyncio
# import os
# import together
# from together import AsyncTogether, Together

import asyncio
import os, sys

from async_client import AsyncClient

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from openai import OpenAI, AsyncOpenAI, AsyncAzureOpenAI, AzureOpenAI, RateLimitError
from dotenv import load_dotenv

load_dotenv()
reference_models = {
    "Llama-3.3-70B-Instruct": {
    "base_url": "https://sisuaidc-agent.services.ai.azure.com/openai/v1/",
    "api_key": os.environ.get("AzureAI_key")
},
    "qwen2.5-72b-instruct": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": os.environ.get("Qwen_API_Key")
    },
    "qwen2.5-coder-32b-instruct": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": os.environ.get("Qwen_API_Key")
    },
    "deepseek-chat": {
        "base_url": "https://api.deepseek.com",
        "api_key": os.environ.get("DeepSeek_API_Key")
    }}

async_clients = AsyncClient(reference_models)

aggregator_model = "deepseek-chat"
aggreagator_system_prompt = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""
client = OpenAI(api_key=os.environ.get("DeepSeek_API_Key"), base_url=reference_models["deepseek-chat"]["base_url"])
user_prompt = "What are 3 fun things to do in SF?"
layers = 2


def getFinalSystemPrompt(system_prompt, results):
    """Construct a system prompt for layers 2+ that includes the previous responses to synthesize."""
    return (
            system_prompt
            + "\n"
            + "\n".join([f"{i + 1}. {str(element)}" for i, element in enumerate(results)])
    )


async def run_llm(model, prev_response=None):
    """Run a single LLM call with a model while accounting for previous responses + rate limits."""
    # print("Final System Prompt: ", getFinalSystemPrompt(aggreagator_system_prompt, prev_response) if prev_response else None)
    for sleep_time in [1, 2, 4]:
        try:
            messages = (
                [
                    {
                        "role": "system",
                        "content": getFinalSystemPrompt(
                            aggreagator_system_prompt, prev_response
                        ),
                    },
                    {"role": "user", "content": user_prompt},
                ]
                if prev_response
                else [{"role": "user", "content": user_prompt}]
            )

            response = await async_clients.acall(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=512,
            )
            print("Model: ", model, "Response: ", response)
            break
        except RateLimitError as e:
            print(e)
            await asyncio.sleep(sleep_time)
    return response


async def main():
    """Run the main loop of the MOA process."""
    results = await asyncio.gather(*[run_llm(model) for model in reference_models])

    # 这里，中间层的llm在回答的时候，每个llm也参考了前一层的四个reference model的答案，那也做了aggregator？
    for _ in range(1, layers - 1):
        results = await asyncio.gather(
            *[run_llm(model, prev_response=results) for model in reference_models]
        )

    finalStream = client.chat.completions.create(
        model=aggregator_model,
        messages=[
            {
                "role": "system",
                "content": getFinalSystemPrompt(aggreagator_system_prompt, results),
            },
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    )
    for chunk in finalStream:
        print(chunk.choices[0].delta.content or "", end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
