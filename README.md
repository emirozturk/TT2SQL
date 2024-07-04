# TT2SQL

This repository contains the datasets used for training (TrainData) and testing (test.jsonl) the Turkish SQL dataset, as well as modified codes.

## Training
Training was conducted using the [SuperAdapters](https://github.com/cckuailong/SuperAdapters) repository.

## Inference
The `InferenceBatch` file requires the SuperAdapters library to run. The `InferenceBatch` file is a modified version of the inference code from the SuperAdapters repository to obtain batch results.

## Models and Links
The fine-tuned models and their links are as follows:

| Model                   | Link                                                        |
|-------------------------|-------------------------------------------------------------|
| Llama 2 SQL             | [emirozturk/Llama-2-7b-chat-hf-SQL](https://huggingface.co/emirozturk/Llama-2-7b-chat-hf-SQL)          |
| Llama 2 Turkish         | [emirozturk/Llama-2-7b-chat-hf-Turkish](https://huggingface.co/emirozturk/Llama-2-7b-chat-hf-Turkish)   |
| Llama 2 Turkish SQL     | [emirozturk/Llama-2-7b-chat-hf-Turkish-SQL](https://huggingface.co/emirozturk/Llama-2-7b-chat-hf-Turkish-SQL) |
| Llama 3 SQL             | [emirozturk/llama3-Instruct-SQL](https://huggingface.co/emirozturk/llama3-Instruct-SQL)           |
| Llama 3 Turkish         | [emirozturk/llama3-Instruct-Turkish](https://huggingface.co/emirozturk/llama3-Instruct-Turkish)    |
| Llama 3 Turkish SQL     | [emirozturk/llama3-Instruct-Turkish-SQL](https://huggingface.co/emirozturk/llama3-Instruct-Turkish-SQL) |
| Phi 3 SQL               | [emirozturk/Phi-3-mini-4k-instruct-SQL](https://huggingface.co/emirozturk/Phi-3-mini-4k-instruct-SQL)    |
| Phi 3 Turkish           | [emirozturk/Phi-3-mini-4k-instruct-Turkish](https://huggingface.co/emirozturk/Phi-3-mini-4k-instruct-Turkish) |
| Phi 3 Turkish SQL       | [emirozturk/Phi-3-mini-4k-instruct-Turkish-SQL](https://huggingface.co/emirozturk/Phi-3-mini-4k-instruct-Turkish-SQL) |

* `InferenceBatch` file is a modified version of the inference code from the SuperAdapters repository to obtain batch results.
* `calculateExAndLfAcc.py` calculates Execution Accuracy and LF accuracy values using the Tur2SQL database (data.db) and wikiSQL Evaluation codes.
