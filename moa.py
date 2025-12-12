import asyncio
import os, sys
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from openai import OpenAI, AsyncOpenAI, AsyncAzureOpenAI, AzureOpenAI, RateLimitError
from dotenv import load_dotenv

load_dotenv()
reference_models = [{"model": "Llama-3.3-70B-Instruct",
                     "base_url": "https://sisuaidc-agent.services.ai.azure.com/openai/v1/",
                     "api_key": os.environ.get("AzureAI_key")
                     },
                     {"model": "qwen2.5-72b-instruct",
                      "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                      "api_key": os.environ.get("Qwen_API_Key")
                      },
                     {"model": "qwen2.5-coder-32b-instruct",
                      "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                      "api_key": os.environ.get("Qwen_API_Key")
                      },
                     {"model": "gpt-4.1-nano",
                      "base_url": "https://sisuaidc-agent.cognitiveservices.azure.com/",
                      "api_key": os.environ.get("AzureAI_key")
                      }]
aggregator_model = "qwen2.5-72b-instruct"
aggreagator_system_prompt = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability."""
async_Llama_client = AsyncOpenAI(api_key=reference_models[0]["api_key"], base_url=reference_models[0]["base_url"])
async_client = AsyncOpenAI(api_key=reference_models[1]["api_key"], base_url=reference_models[1]["base_url"])
async_azure_client = AsyncAzureOpenAI(
        api_version="2024-12-01-preview", # 这个api_version一定需要嘛？
        api_key=reference_models[3]["api_key"],
        azure_endpoint=reference_models[3]["base_url"],
    )
client = OpenAI(api_key=os.environ.get("Qwen_API_Key"), base_url=reference_models[1]["base_url"])

user_prompt = "What are 3 fun things to do in SF?"
async def run_llm(model):
    """Run a single LLM call with a reference model."""
    for sleep_time in [1, 2, 4]:
        try:
            if model == "gpt-4.1-nano":
                response = await async_azure_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=0.7,
                    max_tokens=128,
                )
            elif model == "Llama-3.3-70B-Instruct":
                response = await async_Llama_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=0.7,
                    max_tokens=128,
                )
            else:
                response = await async_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=0.7,
                    max_tokens=128,
                )
            break
        except RateLimitError as e:
            print(e)
            await asyncio.sleep(sleep_time)
    return response.choices[0].message.content

async def main():
    results = await asyncio.gather(*[run_llm(model_dict["model"]) for model_dict in reference_models])
    finalStream = client.chat.completions.create(
        model=aggregator_model,
        messages=[
            {"role": "system", "content": aggreagator_system_prompt + "\n" + "\n".join([f"{i+1}. {str(element)}" for i, element in enumerate(results)])},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    )
    for chunk in finalStream:
        print(chunk.choices[0].delta.content or "", end="", flush=True)





if __name__ == "__main__":

    asyncio.run(main())