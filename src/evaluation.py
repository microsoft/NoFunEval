import os
import csv
import time 
import argparse
import jsonlines
import pandas as pd
from statistics import mean
from jinja2 import Environment, FileSystemLoader
from utils import pass_at_k_continuous_vals, diff_bleu, post_process_generations, statistical_significance_test, remove_comments, remove_blank_lines,get_files_with_syntax_errors

parser = argparse.ArgumentParser()
parser.add_argument("--data_subset", type = str, default = "latency", help = "latency/resource_util/runtime_efficiency/maintenance/security")
parser.add_argument("--model", type = str, default = "wizardcoder", help = "model name")
parser.add_argument("--model_path", type = str, required = True, help = "HF path for OS models")
parser.add_argument("--prompt", type = str, default = "base_prompt", help = "base_prompt/coding_concepts/chain_of_thought/one-shot")
parser.add_argument("--num_samples", type = int, default = 1, help = "Number of samples")
parser.add_argument("--score_k", type = str, default = "1,5,10,20", help = "K value for score@k (should not be greater than num_samples and can be comma-separated)")
parser.add_argument("--metric", type = str, default = "runtime", help = "runtime/diffbleu/codeql-diffbleu")
args = parser.parse_args()

args.model = args.model_path.split("/")[-1]

generations_path = os.path.join("generations", "edit", args.data_subset, args.model, args.prompt, f"{args.num_samples}_samples", "generated_outputs.jsonl")

if(args.metric == "classification"):
    
    generations_path = os.path.join("generations", "classification", args.data_subset, os.path.split(args.model_path)[1], args.prompt, f"{args.num_samples}_samples", "generated_outputs.jsonl")

    left_predictions = []
    right_predictions = []
    left_labels = []
    right_labels = []

    with jsonlines.open(generations_path) as reader:

        for generation in reader:

            left_predictions.append(generation['left_output']) 
            left_labels.append(generation['classification_left_label'])       
            right_predictions.append(generation['right_output'])
            right_labels.append(generation['classification_right_label'])       

    left_accuracy = sum([1 if prediction == label else 0 for prediction, label in zip(left_predictions, left_labels)]) / len(left_labels)
    left_consistency = sum([1 if prediction is not None else 0 for prediction in left_predictions]) / len(left_predictions)
    right_accuracy = sum([1 if prediction == label else 0 for prediction, label in zip(right_predictions, right_labels)]) / len(right_labels)
    right_consistency = sum([1 if prediction is not None else 0 for prediction in right_predictions]) / len(right_predictions)

    joint_accuracy = [1 if left_prediction == left_label and 
                        right_prediction == right_label else 0 
                        for left_prediction, left_label, right_prediction, right_label 
                        in zip(left_predictions, left_labels, right_predictions, right_labels)]

    joint_accuracy = sum(joint_accuracy) / len(joint_accuracy)

    result_string = {"Model":args.model, "left_accuracy":round((left_accuracy*100),1), "right_accuracy":round((right_accuracy*100),1), "joint_accuracy":round((joint_accuracy*100),1), "left_consistency":round((left_consistency*100),1), "right_consistency":round((right_consistency*100),1)}

    output_path = os.path.join("evaluation_results","classification",args.data_subset,os.path.split(args.model_path)[1],args.prompt,f"{args.num_samples}_samples")
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    samples = pd.DataFrame([result_string])
    samples.to_json(os.path.join(output_path,"results.jsonl"), orient="records", lines=True) 
    print("{}".format(result_string))


# To calculate runtimes(Applicable for non-func runtime_efficiency)
elif args.metric == "runtime":

    start_time = time.time()
    
    with jsonlines.open(generations_path) as reader:  
    
        samples = []
    
        for generation in reader:
    
            parsed_generations = []

            for l in range(args.num_samples):

                generated_answers = post_process_generations(generated_answers=generation['generated_answers'][l], model = args.model, prompt = args.prompt, pl = generation['pl'])[1]
                parsed_generations.append(generated_answers)

            samples.append(dict(problem_id = generation['problem_id'], submission_id_v0 = generation['submission_id_v0'], cpu_time_v0 = generation['cpu_time_v0'], cpu_time_v1 = generation['cpu_time_v1'], input=generation['source_code'], target=generation['target_code'],
            generated_answers=parsed_generations, inference_time=generation['inference_time']))
    
        samples = pd.DataFrame(samples)
        path = os.path.join("src", "evaluation", "pie-perf", "generated_outputs.jsonl")
        samples.to_json(path, orient="records", lines = True) 
    
    env = Environment(loader = FileSystemLoader(os.path.join("src", "evaluation", "pie-perf", "data", "sample")))
    template = env.get_template('sample_eval_config_template.yaml')
    output_path = os.path.join("evaluation_results", "edit", args.data_subset, args.model, args.prompt, f"{args.num_samples}_samples", "generated_outputs.report")
    rendered_yaml = template.render(output_path = output_path)
    config_file_path = os.path.join("src", "evaluation", "pie-perf", "data", "sample", "sample_eval_config.yaml")
    f=open(config_file_path, "w")
    f.write(rendered_yaml)
    f.close()
    path = os.path.split(output_path)[0]
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    run_file = os.path.join("src", "evaluation", "pie-perf", "src", "codenet_eval", "run_eval.py")
    
    os.system(f'python3 {run_file} --eval_config {config_file_path}')
    k_values = list(map(int,args.score_k.split(",")))
    scores = statistical_significance_test(output_path,args.num_samples,k_values)
    
    results = {"model":args.model, "prompt":args.prompt, "num_samples":args.num_samples}
    
    for i,j in zip(range(2,len(results),3),range(len(k_values))):

        results[f"Average Speedups@{k_values[j]},{args.num_samples}"] = scores[i-2]
        results[f"Correct@{k_values[j]},{args.num_samples}"] = scores[i-1]
        results[f"Improvements@{k_values[j]},{args.num_samples}"] = scores[i]
    
    samples = pd.DataFrame([results])
    samples.to_json(os.path.join(path, "results.jsonl"), orient="records", lines=True) 
    print("{}".format(results))

# To calculate diffbleu(Applicable for all splits)
elif args.metric == "diffbleu":

    k_values = list(map(int,args.score_k.split(",")))
    overall_score={}
    
    for k in k_values:
        overall_score[k] = []
    
    passed = 0
    count = 0
    
    with jsonlines.open(generations_path) as reader:

        for generation in reader:
            
            count += 1                                    
            scores = []

            for l in range(args.num_samples):

                generated_answers = post_process_generations(generated_answers = generation['generated_answers'][l], model = args.model, prompt = args.prompt, pl = generation['pl'])
                passed += generated_answers[0]
                diff_score_bleu = diff_bleu(source_code = generation['source_code'], target = generation['target_code'], generated_answers = generated_answers[1], pl = generation['pl'])
                scores.append(diff_score_bleu)

            scores.sort(reverse = True)

            for k in k_values:

                overall_score[k].append(pass_at_k_continuous_vals(n = args.num_samples, k = k, vals = scores))

    scores = []
    scores.append((passed*100)/(count*args.num_samples))
    results = {"model":args.model, "prompt":args.prompt, "num_samples":args.num_samples}

    for k in k_values:

        results[f"Score@{k},{args.num_samples}"] = round(mean(overall_score[k])*100,1)
        scores.append(round(mean(overall_score[k])*100,1))

    results["Passed"] = (passed*100)/(count*args.num_samples)
    samples = pd.DataFrame([results])
    path = os.path.join("evaluation_results", "edit", args.data_subset, args.model, args.prompt, f"{args.num_samples}_samples")

    if not os.path.exists(path):
        os.makedirs(path)

    samples.to_json(os.path.join(path,"results_{}.jsonl".format(args.metric)), orient="records", lines=True) 
    print("Pass Rate: {}, DiffBleu Score: {}".format(scores[0], scores[1]))

# To run codeql(Applicable for security and maintenance)
elif args.metric == "codeql":

    all_check_paths={}
    query_lang = {}

    with jsonlines.open(generations_path) as reader:
        
        for generation in reader:

            query = generation['codeql_check'].split("/")[-1].split(".ql")[0]

            all_check_paths[query]=generation['codeql_check']

            code_path="evaluation_results/edit/{}/{}/{}/{}_samples/generated_code/{}/".format(args.data_subset, args.model, args.prompt, args.num_samples, query)

            if not os.path.exists(code_path):
                os.makedirs(code_path)

            if(generation['pl'] == "python"):
                ext = ".py"
                pl = "Python"
                query_lang[query] = "python"
            else:
                ext = ".c"
                pl = "C"
                query_lang[query] = "cpp"

            for index in range(len(generation['generated_answers'])):

                code_path_indexed = code_path + "{}_{}{}".format(generation['file_path'].split("/")[-2]+"_"+generation['file_path'].split("/")[-1].split(ext)[0], index, ext)
            
                f = open(code_path_indexed,"w+")

                generated_answers = post_process_generations(generated_answers=generation['generated_answers'][index], model=args.model, prompt=args.prompt, pl=generation['pl'])[1]

                code = remove_comments(generated_answers, generation['pl'])
                
                if remove_blank_lines(code).strip() == "":
                    generated_answers = generation['source_code']

                f.write(generated_answers)

                f.close()

                if(pl == "C"):

                    f = open(code_path+"Makefile", "w+")
                    f.write("SRCS=$(wildcard *.c)\nOBJS=$(SRCS:.c=.o)\n\nall: $(OBJS)\n\n%.o: %.c\n	gcc -g -O -c $< -o $@ || (echo \"Deleting $<\" && echo \"$<\" >> rejected_files.log && mv $< $<.reject)\n\nclean:\n\trm -rf *.o")
                    f.close()

    for query in all_check_paths.keys():
        
        code_path_generations = "evaluation_results/edit/{}/{}/{}/{}_samples/generated_code/".format(args.data_subset, args.model, args.prompt, args.num_samples)

        code_path_db = "evaluation_results/edit/{}/{}/{}/{}_samples/generated_code_db/".format(args.data_subset, args.model, args.prompt, args.num_samples)
        if not os.path.exists(code_path_db):
            os.makedirs(code_path_db)

        code_path_results="evaluation_results/edit/{}/{}/{}/{}_samples/generated_code_results/".format(args.data_subset, args.model, args.prompt, args.num_samples)
        if not os.path.exists(code_path_results):
            os.makedirs(code_path_results)

        os.system("codeql-home/codeql/codeql database create --quiet --language={} --source-root={}{} {}{}".format(query_lang[query], code_path_generations, query, code_path_db, query))
        os.system("codeql-home/codeql/codeql database analyze --rerun {}{} {} --format=csv --output={}{}.csv --threads=0".format(code_path_db, query, all_check_paths[query], code_path_results, query))

    k_values = list(map(int,args.score_k.split(",")))
    overall_score={}
    
    for k in k_values:
        overall_score[k] = []

    syntax_errors = {}
    syn_errors = []
    done = []
    scores_dump = []

    with jsonlines.open(generations_path) as reader:
        
        parsed = 0

        for generation in reader:
            query = generation['codeql_check'].split("/")[-1].split(".ql")[0]
                
            code_path  ="evaluation_results/edit/{}/{}/{}/{}_samples/generated_code/{}/".format(args.data_subset, args.model, args.prompt, args.num_samples, query)
            scores = []
            code_path_results = "evaluation_results/edit/{}/{}/{}/{}_samples/generated_code_results/{}.csv".format(args.data_subset, args.model, args.prompt, args.num_samples, query)
            code_path_generations = "evaluation_results/edit/{}/{}/{}/{}_samples/generated_code/{}/".format(args.data_subset, args.model, args.prompt, args.num_samples, query)
            code_path_db = "evaluation_results/edit/{}/{}/{}/{}_samples/generated_code_db/".format(args.data_subset, args.model, args.prompt, args.num_samples)

            errors=[]

            with open(code_path_results) as f:
                csvfile = csv.reader(f)
                for error in csvfile:
                    errors.append(error[-5].split("/")[1])

            errors = list(set(errors))
            index = 0
            scores = []
            ans = []
            syn=get_files_with_syntax_errors(generated_code_path = code_path_generations, codeql_db_path = code_path_db, query = query)

            if(len(syn)>0 and query not in done):
                
                syn_errors += syn
                try:
                    syntax_errors[query] += syn
                except:
                    syntax_errors[query] = syn
            
            done.append(query)
            
            for index in range(len(generation['generated_answers'])):

                if(generation['pl'] == "python"):
                    ext = ".py"
                    pl = "Python"
                else:
                    ext = ".c"
                    pl = "C"

                filename = "{}_{}{}".format(generation['file_path'].split("/")[-2]+"_"+generation['file_path'].split("/")[-1].split(ext)[0], index, ext)

                index += 1

                if(filename in errors or filename in syn_errors):
                    scores.append(0)
                else:
                    scores.append(1)

            scores.sort(reverse = True)

            for k in k_values:

                overall_score[k].append(pass_at_k_continuous_vals(n = args.num_samples, k = k, vals = scores))

            scores_dump.append(scores)
            scores=[]

    path = os.path.join("evaluation_results", "edit", args.data_subset, args.model, args.prompt, f"{args.num_samples}_samples")
    f = open(os.path.join(path,"results.txt"), 'w')
    f.write(str(scores_dump))
    f.close()
    results = {"model":args.model, "prompt":args.prompt, "num_samples":args.num_samples}

    for k in k_values:

        results[f"Score@{k},{args.num_samples}"] = round(mean(overall_score[k])*100,1)

    results["syntax_errors"] = syntax_errors
    results["no_of_syntax"] = len(syn_errors)
    samples = pd.DataFrame([results])
    path = os.path.join("evaluation_results", "edit", args.data_subset, args.model, args.prompt, f"{args.num_samples}_samples")
    samples.to_json(os.path.join(path, "results_{}.jsonl".format(args.metric)), orient="records", lines=True) 
    print("{}".format(results))
    
 

# get codeql*diffbleu numbers(Applicable for security and maintenance)
elif args.metric == "codeql-diffbleu":
    k_values = list(map(int, args.score_k.split(",")))
    overall_score = {}
    
    for k in k_values:
        overall_score[k] = []

    generations_path = os.path.join("generations","edit", args.data_subset, args.model, args.prompt, f"{args.num_samples}_samples", "generated_outputs.jsonl")
    
    passed = 0
    count = 0       

    with jsonlines.open(generations_path) as reader:

        res_path = os.path.split(generations_path)[0].split('/')
        res_path.insert(1,"evaluation_results")
        res_path = os.path.join("/".join(res_path[1:]), "results.txt")
        codeql_results = eval(open(res_path).read())

        for generation,res in zip(reader, codeql_results):

            scores=[]
            count += 1

            for l in range(len(generation['generated_answers'])):

                generated_answers=post_process_generations(generated_answers = generation['generated_answers'][l], model = args.model, prompt = args.prompt, pl = generation['pl'])
                passed += generated_answers[0]
                diff_score_bleu=res[l]*diff_bleu(source_code = generation['source_code'], target = generation['target_code'], generated_answers = generated_answers[1], pl = generation['pl'])
                scores.append(diff_score_bleu)

            scores.sort(reverse = True)

            for k in k_values:
                overall_score[k].append(pass_at_k_continuous_vals(n = args.num_samples, k = k, vals = scores))

    scores = []
    scores.append((passed*100)/(count*args.num_samples))
    results = {"model":args.model, "prompt":args.prompt, "num_samples":args.num_samples}
    
    for k in k_values:
    
        results[f"Score@{k},{args.num_samples}"] = round(mean(overall_score[k])*100,1)
        scores.append(round(mean(overall_score[k])*100,1))
    
    results["Passed"] = (passed*100)/(count*args.num_samples)
    scores.append((passed*100)/(count*args.num_samples))
    samples = pd.DataFrame([results])
    path = os.path.join("evaluation_results", "edit", args.data_subset, args.model, args.prompt, f"{args.num_samples}_samples")
    samples.to_json(os.path.join(path,"results_{}.jsonl".format(args.metric)), orient="records", lines=True) 
    print("{}".format(results))