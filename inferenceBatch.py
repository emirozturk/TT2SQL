#Modified inference file from SuperAdapters repo for batch processing
#Needs SuperAdapters repo to run and generate results

import os
import sys
import time
import json

import gradio as gr

from core.seq2seq.chatglm import ChatGLMSeq2Seq
from core.seq2seq.llama import LLAMASeq2Seq
from core.seq2seq.bloom import BLoomSeq2Seq
from core.seq2seq.qwen import QwenSeq2Seq
from core.seq2seq.baichuan import BaichuanSeq2Seq
from core.seq2seq.mixtral import MixtralSeq2Seq
from core.seq2seq.phi import PhiSeq2Seq
from core.seq2seq.gemma import GemmaSeq2Seq

from core.classify.llama import LLAMAClassify
from core.classify.bloom import BLoomClassify

from api.app import create_app
from tqdm import tqdm


paramsList = [
        {
        "model_name": "llama2-tr",
        "model_type": "llama2",
        "model_path": "meta-llama/Llama-2-7b-chat-hf",
        "adapter_weights": "Llama-2-7b-chat-hf-Turkish-SQL",
        "max_new_tokens": "32"
    },
    {
        "model_name": "llama3-base",
        "model_type": "llama3",
        "model_path": "meta-llama/Meta-Llama-3-8B-Instruct",
        "adapter_weights": "llama3-Instruct",
        "max_new_tokens": "32"
    },
    {
        "model_name": "llama3-tr",
        "model_type": "llama3",
        "model_path": "meta-llama/Meta-Llama-3-8B-Instruct",
        "adapter_weights": "llama3-Instruct-Turkish-SQL",
        "max_new_tokens": "32"
    },
    {
        "model_name": "llama2-base",
        "model_type": "llama2",
        "model_path": "meta-llama/Llama-2-7b-chat-hf",
        "adapter_weights": "Llama-2-7b-chat-hf",
        "max_new_tokens": "32"
    },
    {
        "model_name": "phi3-base",
        "model_type": "phi3",
        "model_path": "microsoft/Phi-3-mini-4k-instruct",
        "adapter_weights": "Phi-3-mini-4k-instruct",
        "max_new_tokens": "32"
    },
    {
        "model_name": "phi3-tr",
        "model_type": "phi3",
        "model_path": "microsoft/Phi-3-mini-4k-instruct",
        "adapter_weights": "Phi-3-mini-4k-instruct-Turkish-SQL",
        "max_new_tokens": "32"
    }
]

# Read instructions from the test.jsonl file
instructions = []
with open('test.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        instruction_data = json.loads(line)
        instructions.append(instruction_data)

if __name__ == "__main__":
    for params in paramsList:
        model_type = params["model_type"]
        if model_type == "chatglm" or model_type == "chatglm2":
            llm = ChatGLMSeq2Seq()
        elif model_type == "llama" or model_type == "llama2" or model_type == "llama3":
            llm = LLAMASeq2Seq()
        elif model_type == "bloom":
            llm = BLoomSeq2Seq()
        elif model_type == "qwen":
            llm = QwenSeq2Seq()
        elif model_type == "baichuan":
            llm = BaichuanSeq2Seq()
        elif model_type == "mixtral":
            llm = MixtralSeq2Seq()
        elif model_type == "phi" or model_type == "phi3":
            llm = PhiSeq2Seq()
        elif model_type == "gemma":
            llm = GemmaSeq2Seq()
        else:
            print("model_type should be llama/llama2/llama3/bloom/chatglm/chatglm2/qwen/baichuan/mixtral/phi/phi3/gemma")
            sys.exit(-1)
    
        llm.debug = False
        llm.web = False

        llm.base_model = params["model_path"]
        llm.model_type = params["model_type"]
        llm.adapter_weights = params["adapter_weights"]

        llm.load_8bit = False

        llm.temperature = 0.7
        llm.top_p = 0.9
        llm.top_k = 40
        llm.max_new_tokens = 32 #For Phi generation, it must be > 80 for inteference
        llm.device = "cpu"

        output_file = f'{params["model_name"]}-predictions.jsonl'
        
        # Check for existing predictions to resume
        processed_instructions = set()
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as outfile:
                for line in outfile:
                    existing_data = json.loads(line)
                    processed_instructions.add(existing_data["instruction"])
        
        model = llm.load_model()
        with open(output_file, 'a', encoding='utf-8') as outfile:
            for instruction_data in tqdm(instructions):
                instruction = instruction_data['instruction']
                output = instruction_data['output']
                
                if instruction in processed_instructions:
                    continue
                
                start = time.time()
                prediction = llm.generate(model,instruction, None, None, False, None, None, None)
                end = time.time()
                print(f"Eval Cost: {end-start} seconds")
                
                result = {
                    "instruction": instruction,
                    "output": output,
                    "prediction": prediction
                }
                json.dump(result, outfile)
                outfile.write('\n')