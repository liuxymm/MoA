#!/bin/bash

# 检查操作系统
#if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
#    # Windows Git Bash
#    PYTHON="D:/project/MoA/venv/Scripts/python.exe"
#    SEP="/"
#else
    # Linux/Mac
#    PYTHON="./venv/bin/python"
#    SEP="/"
#fi

export DEBUG=1

# reference_models="microsoft/WizardLM-2-8x22B,Qwen/Qwen1.5-110B-Chat,Qwen/Qwen1.5-72B-Chat,meta-llama/Llama-3-70b-chat-hf,mistralai/Mixtral-8x22B-Instruct-v0.1,databricks/dbrx-instruct"
reference_models="Llama-3.3-70B-Instruct,qwen2.5-72b-instruct,qwen2.5-coder-32b-instruct,deepseek-chat"

#python generate_for_alpaca_eval.py \
#    --model="qwen2.5-72b-instruct" \
#    --output-path="outputs/qwen2.5-72b-instruct.json" \
#    --reference-models=${reference_models} \
#    --rounds 3 \
#    --num-proc 3

python generate_for_alpaca_eval.py \
    --model="qwen2.5-72b-instruct" \
    --output-path="outputs" \
    --reference-models=${reference_models} \
    --rounds 3 \
    --num-proc 1


alpaca_eval --model_outputs outputs/qwen2.5-72b-instruct/checkpoints/checkpoint_32.json --reference_outputs alpaca_eval/results/gpt4_1106_preview/model_outputs.json --output_path leaderboard


#nohup sh run_eval_alpaca_eval.sh >> output1.log 2>&1 &
# 终止：
# 1. 查找所有相关进程（python和sh）
#ps aux | grep -E "generate_for_alpaca_eval|run_eval_alpaca_eval.sh" | grep -v grep
#
## 2. 同时kill找到的所有进程
#pkill -f "generate_for_alpaca_eval"
#pkill -f "your_script.sh"  # 你的sh脚本文件名
#
## 或一条命令搞定
#pkill -f -e "generate_for_alpaca_eval|run_eval_alpaca_eval.sh"



# 最终输出保存在： outputs/qwen2.5-72b-instruct/checkpoints/checkpoint_32.json