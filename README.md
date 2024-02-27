# Introduction
This repository hosts the official code and data artifact for the paper ["NoFunEval: Funny How Code LMs Falter on Requirements Beyond Functional Correctness"](https://arxiv.org/abs/2401.15963). The work is a comprehensive evaluation of code language models on real-world code editing scenarios and metrics beyond functional correctness.

## Repository Contents
1. [Datasets](#1-datasets): Runtime Efficiency, Maintainability, Latency, Resource Utilization, Security, HumanEvalClassify
3. [Generations](#2-generations): Generated code for examples in NoFunEval with various model configurations reported in the paper. The graphs and tables reported in the paper can be reproduced by running the evaluation scripts on the provided generations. One can also generate code using a new model using the script. 
2. [Evaluation scripts](#3-evaluation-scripts): Scripts to evaluate LMs by taking input from examples in NoFunEval and producing score@k scores for the metrics reported in the paper: DiffBleu, Average SpeedUp, CodeQL, CodeQL-DiffBleu.

## Datasets

# Generation
## Environment Setup
Create a virtual environment. 
```console
bash setup.sh
```

### NoFunEdit
```console
python3 src/nofunedit_generation.py --data_subset <subset from nofunedit: eg-latency> --model_path <model name from HF: eg-WizardLM/WizardCoder-15B-V1.0> --temperature <temperature to be set for model generation: eg-0> --max_new_tokens <maximum number of new tokens to be generated: eg-5192> --prompt <type of prompt to use from our dataset: eg-base_prompt> --num_samples <number of samples to be generated: eg-1> --precision <floating point format: eg-fp16> --batch_size <number of examples to send to llm engine at once: eg-1>
```
### Classification
```console
python3 src/classification_generation.py --data_subset <subset from non_func or humanevalclassify: eg-latency> --model <model name from HF: eg-WizardLM/WizardCoder-15B-V1.0> --temperature <temperature to be set for model generation: eg-0> --max_new_tokens <maximum number of new tokens to be generated: eg-5192> --prompt <type of prompt to use from our dataset: eg-base_prompt> --precision <floating point format: eg-fp16> --batch_size <number of examples to send to llm engine at once: eg-1>
```
# Evaluation Scripts

## Evaluation
```console
python3 src/evaluation.py --data_subset <subset from nofunedit: eg-latency> --model_path <model name from HF: eg-WizardLM/WizardCoder-15B-V1.0> --prompt <type of prompt to use from our dataset: eg-base_prompt> --num_samples <number of samples to be generated: eg-1> --score_k <K values for score@k: eg-1,5,10,20> --metric <eval_metric to be used: eg-diffbleu>
```

### Example eval script (For maintainability)
```console
bash evaluation_example_script.sh
```

## Parameters

| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `data_subset`                     | The subset of data to use. Options: `latency`, `resource_util`, `maintainability`, `security`, `runtime_efficiency` for nofunedit. Additionally `humanevalclassify` for classification.|
| `model_path` | The path of the model from HF. Example: `WizardLM/WizardCoder-15B-V1.0`.
| `prompt`      | Prompt to use. Options: `base_prompt`, `one-shot`, `chain_of_thought`, `coding_concepts`. |
| `num_samples` | Number of samples to generate. Example: `1` (We used  `1` for greedy and `20` for higher temperature). **[NoFunEdit - Generation only]**|
| `max_new_tokens` | Budget for new token generation for a model. Example: `1200` (NoFunEdit: We used `1200` for runtime_efficiency and security for all prompts than CoT where `1500` was used. For others, we used `5192` or max possible limit. Classification: We used `4` for all generations).|
| `temperature` | Temperature for model generation. Example: `0` (We used `0` for greedy and `0.8` for higher samples) |
| `score_k` |K vales for Score@K. Example: `1,5,10,20` (Should not be greater than num_samples and is comma separated)  **[Eval only]** |
| `metric` | Metric to be used for evaluation. Option:  `diffbleu`, `codeql`, `codeql-diffbleu` (to be run after first two params are run), `classification`, `runtime` **[Eval only]**|

#### VLLM Parameters (for generation)
| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `batch-size` | Batch size. Default: `1`|
| `precision` | Floating point format: Default: `fp16` |
|  `tensor_parallel_size` | Default: `1` |
| `swap_space` | The size (GiB) of CPU memory per GPU to use as swap space: Default: `4` |

## Code Reference
Our evaluation code for runtime efficiency has been derived from the [PIE codebase](https://github.com/madaan/pie-perf).

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.
