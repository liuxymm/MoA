


    # # Qwen 72B
    # client = OpenAI(api_key=os.environ.get("Qwen_API_Key"), base_url=reference_models[1]["base_url"])
    # completion = client.chat.completions.create(
    #     # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    #     model=reference_models[1]["model"],
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": "你是谁？"},
    #     ]
    # )
    # print(completion.model_dump_json())
    #
    # # AzureAI-Llama-3.3-70B-Instruct
    # client = OpenAI(
    #     base_url=reference_models[0]["base_url"],
    #     api_key=os.environ.get("AzureAI_key")
    # )
    # completion = client.chat.completions.create(
    #     model=reference_models[0]["model"],
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": "What is the capital of France?",
    #         }
    #     ],
    # )
    #
    # print(completion.choices[0].message)
    #
    #
    # #gpt-4.1-nano
    # deployment = reference_models[3]["model"]
    # azure_client = AzureOpenAI(
    #     api_version="2024-12-01-preview",
    #     api_key=os.environ.get("AzureAI_key"),
    #     azure_endpoint=reference_models[3]["base_url"],
    # )
    # response = azure_client.chat.completions.create(
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": "You are a helpful assistant.",
    #         },
    #         {
    #             "role": "user",
    #             "content": "I am going to Paris, what should I see?",
    #         }
    #     ],
    #     max_completion_tokens=13107,
    #     temperature=1.0,
    #     top_p=1.0,
    #     frequency_penalty=0.0,
    #     presence_penalty=0.0,
    #     model=deployment
    # )
    # print(response.choices[0].message.content)