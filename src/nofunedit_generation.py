import os
import time
import argparse
import jsonlines
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


#Input all the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_subset", type = str, default = "latency", help = "type of non-func requirement")
parser.add_argument("--temperature", type = float, default = 0.0, help = "temperature")
parser.add_argument("--max_new_tokens", type = int, default = 5192, help = "max length of tokens")
parser.add_argument("--top_p", type = float, default = 0.95, help = "top_p")
parser.add_argument("--prompt", type = str, default = "base_prompt", help = "type of prompt")
parser.add_argument("--num_samples", type = int, default = 1, help = "number of samples")
parser.add_argument("--model_path", type = str, required=True, help = "HF path for OS models")
parser.add_argument("--load_in_8bit", action="store_true", help = "Load model in 8bit")
parser.add_argument("--load_in_4bit", action="store_true", help = "Load model in 4bit")
parser.add_argument("--precision", type = str, default = "fp16", help = "Model precision, from: fp32, fp16 or bf16")
parser.add_argument("--tensor_parallel_size", type = int, default = 1, help = "Tensor parallel size")
parser.add_argument("--swap_space", type = int, default = 4, help = "The size (GiB) of CPU memory per GPU to use as swap space.")
parser.add_argument("--batch_size", type = int, default = 1, help = "Number of examples to send to llm engine at once.")
args = parser.parse_args()
argsdict = vars(args)


def model_query(all_messages, batch_size = 1):

    all_messages = [messages[0]["content"] for messages in all_messages]
    
    llm_tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        truncation_side="left",
        padding_side="right",  # padding on the right is needed to cut off padding in `complete_code`
    )
    
    if args.num_samples == 1:
        GREEDY = True
    else:
        GREEDY = False
    
    assert args.num_samples % batch_size == 0, "num_samples must be divisible by batch_size"
    
    sampling_params = SamplingParams(
                                    n = batch_size, # for multisamples we sample multiple times
                                    temperature = args.temperature if not GREEDY else 0.0, 
                                    top_p = args.top_p if not GREEDY else 1.0,
                                    top_k = 50 if not GREEDY else -1,
                                    max_tokens = args.max_new_tokens,
                                    stop_token_ids = [llm_tokenizer.eos_token_id])
    
    llm = LLM(model = args.model_path,
            tensor_parallel_size = args.tensor_parallel_size,
            swap_space = args.swap_space)

    start_time = time.time()

    for turn_id in tqdm(range(0, args.num_samples//batch_size)):

        llm_outputs = llm.generate(all_messages, sampling_params)

        if turn_id == 0:
            all_generated_answers = [[llm_output.prompt + llm_gen.text 
                                    for llm_gen in llm_output.outputs]   
                                    for llm_output in llm_outputs]
        else:
            for idx, llm_output in enumerate(llm_outputs):
                all_generated_answers[idx].extend([llm_output.prompt + llm_gen.text 
                                                for llm_gen in llm_output.outputs])
                
    total_time = time.time() - start_time
    avg_times = [total_time / len(all_messages)] * len(all_messages)
    
    return all_generated_answers, avg_times


dataset_path = os.path.join("datasets", f"{args.data_subset}.jsonl")

data = []
max_tokens = []
generations = []
all_messages = []

with jsonlines.open(dataset_path) as data_file:

    for data_item in data_file:

        data.append(data_item)
        content = data_item[args.prompt]
        messages=[{"role": "user", "content": content}]
        all_messages.append(messages)

print("Starting model inference...")
all_generated_answers, all_inference_times = model_query(all_messages = all_messages, batch_size = args.batch_size)    

for i, data_item in tqdm(enumerate(data)):

    generated_answers = all_generated_answers[i]
    inference_time = all_inference_times[i]
    curr_sample = data_item
    curr_sample["inference_time"] = inference_time
    curr_sample["generated_answers"] = generated_answers

    for prompt in ["base_prompt", "coding_concepts", "chain_of_thought", "one_shot"]:
        del curr_sample[prompt]
    
    generations.append(curr_sample)   
    
generations = pd.DataFrame(generations)
path = os.path.join("generations", "edit", args.data_subset, os.path.split(args.model_path)[1], args.prompt, f"{args.num_samples}_samples")
if not os.path.exists(path):
    os.makedirs(path)
path=os.path.join(path, "generated_outputs.jsonl")
generations.to_json(path, orient = "records", lines=True)
