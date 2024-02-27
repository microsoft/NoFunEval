import re 
import os 
import sys
import math
import tokenize 
import tiktoken
import tempfile
import jsonlines
import subprocess
import scipy.stats
from io import StringIO
from statistics import mean
from tree_sitter import Language, Parser
from nltk.tokenize import wordpunct_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def pass_at_k_continuous_vals(n, k, vals):

    #Score@k,n is the continuous value version of pass@k,n defined in MGCoder paper.

    assert len(vals) == n
    assert n >= k, (n, k)
    assert all(vals[i-1] >= vals[i] for i in range(1, len(vals))), (n, k, vals)

    isum = 0
    for i in range(1, n-k+2):
        # i ranges from 1 to n-k+1
        isum += (vals[i-1]*math.comb(n-i,k-1))

    return isum/math.comb(n,k)


def remove_comments(code: str, language: str) -> str:

    #Using re module to remove comments for specific languages
    if language.lower() == "python":
        try:
            return remove_py_comments(code)
        except:
            pattern = r'\'{3}[\s\S]*?\'{3}|\"{3}[\s\S]*?\"{3}|\#[^\n]*'

    elif language in ["java", "javascript", "scala", "kotlin", "c++", "c", "ino", "objectivec"]:
        pattern = r"\/\*[\s\S]*?\*\/|\/\/[^\n]*"

    elif language == 'assembly':
        pattern = r';.*|\#.*|\/*[\s\S]*?\*\/'

    elif language == 'javascript xml':
        pattern = r"\/\*[\s\S]*?\*\/|\/\/[^\n]*|<!--.*?-->"

    code=re.sub(pattern, '', code)
    return code    

def remove_blank_lines(code) -> str:

    #Remove blank lines in a string
    try:
        lines = code.split("\n")
        non_blank_lines = [line for line in lines if line.strip() != ""]
        return "\n".join(non_blank_lines)
    except:
        return code

def diff_bleu(source_code, target, generated_answers, pl):

    """Calculating the DiffBleu score.
    It is the bleu score between the git diff of the source and generated code and git diff of the source and target."""

    with tempfile.NamedTemporaryFile(mode = 'w', delete = False) as source_temp, tempfile.NamedTemporaryFile(mode = 'w', delete = False) as target_temp, tempfile.NamedTemporaryFile(mode = 'w', delete = False) as generated_temp:

        source_temp.write(remove_blank_lines(remove_comments(source_code,pl.lower())))
        target_temp.write(remove_blank_lines(remove_comments(target,pl.lower())))
        generated_temp.write(remove_blank_lines(remove_comments(generated_answers,pl.lower())))
        
        source_path = source_temp.name
        target_path = target_temp.name
        generated_answers_path = generated_temp.name

    command_diff_generated = "git diff -U0 --no-index --ignore-all-space --ignore-blank-lines {} {} | tail -n +5 | grep -v 'No newline at end of file'".format(source_path,generated_answers_path)
    command_diff_target = "git diff -U0 --no-index --ignore-all-space --ignore-blank-lines {} {} | tail -n +5 | grep -v 'No newline at end of file'".format(source_path,target_path)
    
    diff_generated = subprocess.run(command_diff_generated, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode()
    diff_target = subprocess.run(command_diff_target, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode()

    diff_generated = wordpunct_tokenize(diff_generated)
    diff_target = wordpunct_tokenize(diff_target)

    diff_score_bleu = sentence_bleu([diff_target], diff_generated, smoothing_function=SmoothingFunction().method1)
    
    if remove_blank_lines(remove_comments(generated_answers, pl.lower())).strip() == "":
        return 0

    return diff_score_bleu

def get_welch_t_test_p(m1: float, s1: float, m2: float, s2: float, n1: int, n2: int) -> float:
    """Returns the p-value of a Welch's t-test. The null hypothesis is that the two samples have the same mean.
    Alternative hypothesis is that the first sample has a smaller mean than the second sample.
    The first distribution is (m1, s1) and the second distribution is (m2, s2). The number of samples in each distribution is n1 and n2 respectively.

    Returns:
        float: p-value
    """
    return scipy.stats.ttest_ind_from_stats(
        mean1 = m1,
        mean2 = m2,
        std1 = s1,
        std2 = s2,
        nobs1 = n1,
        nobs2 = n2,
        equal_var = False,  # Welch's t-test - unequal variances
        alternative = "less",  # one-sided - generated is faster
    ).pvalue

def num_tokens_from_string(string: str, model: str) -> int:

    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def statistical_significance_test(output_path,n,k_values):

    avg_speedup = {} 
    fun_correct = {}
    imp = {}

    for k in k_values:

        avg_speedup[k] = []
        fun_correct[k] = []
        imp[k] = []

    with jsonlines.open(output_path) as f:
        
        for item in f:

            prob_id = item['problem_id']
            scores = []
            correct = []
            sig = []

            for i in range(n):
            
                if item[f"generated_answers_{i}_acc"] != 1:
                    scores.append(1)
                    correct.append(0)
                    sig.append(0)
                else:
                    m1 = item[f'generated_answers_{i}_time_mean']
                    s1 = item[f'generated_answers_{i}_time_std']
                    nobs1 = 25
                    m2 = item['input_time_mean']
                    s2 = item['input_time_std']
                    nobs2 = 25
                    welch = get_welch_t_test_p(m1,s1,m2,s2,nobs1,nobs2)

                    if(welch<.05 and item['input_time_mean'] > item[f'generated_answers_{i}_time_mean']):
                        scores.append(item['input_time_mean']/item[f'generated_answers_{i}_time_mean'])
                        sig.append(1)
                    else:
                        scores.append(1)
                        sig.append(0)
                    correct.append(1)

            scores = sorted(scores,reverse = True)
            correct = sorted(correct,reverse = True)
            sig = sorted(sig,reverse = True)

            for k in k_values:

                avg_speedup[k].append(pass_at_k_continuous_vals(n, k, scores))
                fun_correct[k].append(pass_at_k_continuous_vals(n, k, correct))
                imp[k].append(pass_at_k_continuous_vals(n, k, sig))
 
    scores = []

    for k in k_values:

        scores.append(round(mean(avg_speedup[k]), 3))
        scores.append(round(mean(fun_correct[k]), 3))
        scores.append(round(mean(imp[k]), 3))

    return scores
    
def get_files_with_syntax_errors(generated_code_path, codeql_db_path, query):

    #Find the files with syntax errors on running codeql by finding warning pattern in the db logs created. 
    error_files = []
    error_pattern = r"\[WARN\] \[\d*\] Failed to analyse imports of ([a-zA-Z0-9\\/.:_\-\(\)\']*) : Syntax Error \(line \d*\)"
    error_pattern_expr = re.compile(error_pattern)
    parent_path = os.path.abspath(generated_code_path + "/")
    log_dir = codeql_db_path + "/{}/log/".format(query)
    
    if not os.path.exists(log_dir):
        raise FileNotFoundError(log_dir)
    
    for p in os.listdir(log_dir):
        if p.startswith('database-create'):
            log_path = p
    
    with open(log_dir + log_path, 'r') as f:
        logs = f.read()

    files_with_syntax_error = error_pattern_expr.findall(logs)
    
    for file_with_error in files_with_syntax_error:

        if not os.path.exists(file_with_error):
            print("danger: ", file_with_error)
            sys.exit(-1)
        
        child_path = os.path.abspath(file_with_error)

        if os.path.commonpath([parent_path]) == os.path.commonpath([parent_path, child_path]):
            error_files.append(file_with_error.split("/")[-1])
    
    return error_files

def remove_py_comments(source):

    #Robustly remove python specific comments
    io_obj = StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0

    for tok in tokenize.generate_tokens(io_obj.readline):

        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        ltext = tok[4]
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))
        if token_type == tokenize.COMMENT:
            pass
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                if prev_toktype != tokenize.NEWLINE:
                    if start_col > 0:
                        out += token_string
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line

    return out

def check_syntax(code,language):

    #Checks if the code passed parses without any errors using tree-sitter
    code += '\n'
    
    if(language.lower()  == "java"):
        path = 'src/evaluation/tree-sitter/tree-sitter-java'
    elif(language.lower() == "python"):
        path = 'src/evaluation/tree-sitter/tree-sitter-python'
    elif(language.lower() == "scala"):
        path = 'src/evaluation/tree-sitter/tree-sitter-scala'
    elif(language.lower() == "c"):
        path = 'src/evaluation/tree-sitter/tree-sitter-c'
    elif(language.lower() == "c++"):
        path = 'src/evaluation/tree-sitter/tree-sitter-cpp'
    elif(language.lower() == "objectivec"):
        path = 'src/evaluation/tree-sitter/tree-sitter-objc'
    elif(language.lower() == "javascript"):
        path = 'src/evaluation/tree-sitter/tree-sitter-javascript'
    elif(language.lower() == "kotlin"):
        path = 'src/evaluation/tree-sitter/tree-sitter-kotlin'
    else:
        return(False)

    Language.build_library(
    'src/evaluation/tree-sitter/build/my-languages_{}.so'.format(language.lower()),
    [
        path
    ]
    )

    if(language.lower() == "java"):
        LANGUAGE = Language('src/evaluation/tree-sitter/build/my-languages_java.so', 'java')
    elif(language.lower() == "python"):
        LANGUAGE = Language('src/evaluation/tree-sitter/build/my-languages_python.so', 'python')
    elif(language.lower() == "scala"):
        LANGUAGE = Language('src/evaluation/tree-sitter/build/my-languages_scala.so', 'scala')
    elif(language.lower() == "c"):
        LANGUAGE = Language('src/evaluation/tree-sitter/build/my-languages_c.so', 'c')
    elif(language.lower() == "c++"):
        LANGUAGE = Language('src/evaluation/tree-sitter/build/my-languages_c++.so', 'cpp')
    elif(language.lower() == "objectivec"):
        LANGUAGE = Language('src/evaluation/tree-sitter/build/my-languages_objectivec.so', 'objc')
    elif(language.lower() == "javascript"):
        LANGUAGE = Language('src/evaluation/tree-sitter/build/my-languages_javascript.so', 'javascript')
    elif(language.lower() == "kotlin"):
        LANGUAGE = Language('src/evaluation/tree-sitter/build/my-languages_kotlin.so', 'kotlin')

    parser = Parser()
    parser.set_language(LANGUAGE)
    
    tree = parser.parse(bytes(code, "utf8"))

    def find_error(node):
    
        if node.type == 'ERROR':
            print(f'Error found from line {node.start_point[0]+1}, column {node.start_point[1]+1} to line {node.end_point[0]+1}, column {node.end_point[1]+1}')
        for child in node.children:
            find_error(child)
    
    return not(tree.root_node.has_error)


def extract_parsable_code(start, code, code_list,top,bottom, pl):

    #Checking thw largest parsable combination in a sentence of text
    while top < bottom:

        code1 = " ".join(code_list[top:bottom])
        code2 = " ".join(code_list[top:bottom-1])
        code3 = " ".join(code_list[top+1:bottom])

        if(start):

            if check_syntax(code1 + "\n" + code,pl):
                return code1

            elif check_syntax(code2 + "\n" + code,pl):
                return code2

            elif check_syntax(code3 + "\n" + code,pl):
                return code3
            else:
                top += 1
                bottom -= 1
        
        else:
        
            if check_syntax(code+"\n" + code1,pl):
                return code1

            elif check_syntax(code+"\n" + code2,pl):
                return code2

            elif check_syntax(code+"\n" + code3,pl):
                return code3
            else:
                top += 1
                bottom -= 1

    return None

def post_process_generations(generated_answers: str, model: str, prompt:str, pl: str) -> str:

    """Post processing outputs to first extract the code between backquotes after response as defined in the template.
    Failing which we try to use tree-sitter to obtain maximum parsable block of code but it is the models failure to not follow the tempate
    """

    failed = 0

    generated_answers = remove_blank_lines(generated_answers)

    if(prompt == "one-shot" or prompt == "chain_of_thought"):
        index = 2
    else:
        index = 1

    #Extracting the code after the Response keyword within triple backquotes
    try:
        
        generated_answers=generated_answers.split('Response:')[index].strip().split("Instruction:")[0]

        generated_answers_post=generated_answers.split("```")[1:]

        generated_answers_post = "\n".join("```".join(generated_answers_post).split('\n')[1:])


        generated_pass = 0
        if(generated_answers_post.find("```") != -1):
            generated_answers_post=generated_answers_post.split('```')[0]
            

            passed = 1
            if(generated_answers_post == ""):
                passed = 0
            return [passed,generated_answers_post]
        else:

            failed = 1
    except:        
        if(prompt=="chain_of_thought"):
            try:
                generated_answers="```".join(generated_answers.split("Thought")[2].split("```")[1:])
                if(generated_answers.find("```")!=-1):
                    generated_answers = "\n".join(generated_answers.split("```")[0].split("\n")[1:])
                    return [1,generated_answers]
                else:
                    failed = 1
            except:
                failed = 1
        else:
            failed = 1

    unsupported_pl = ['javascript xml', 'ino', 'assembly']
    
    if(pl.lower() in unsupported_pl):
        generated_answers = remove_blank_lines(generated_answers.strip())
        return [0,generated_answers]

    # If the template is not followed, we try to use tree-sitter to extract parsable code
    if (failed):

        generated_answers = generated_answers.replace('\"','"').replace("\'","'").replace("\/\/","//").replace("\/*","/*").replace("*\/","*/")
        example_list = generated_answers.splitlines()

        for i in range(len(example_list)):
            if(check_syntax(example_list[i],pl)):
                start_index = i
                break
        try:
            start_index
        except:
            start_index = 0

        if len(example_list) == 0:
            return [0,generated_answers]

        example_list[start_index] = example_list[start_index].strip()
        last_index = len(example_list)

        #Finding largest possible codeblock that parses
        while(last_index > start_index):
            code=""
            for j in range(start_index, last_index):
                code += example_list[j]+"\n"
            if(check_syntax(code,pl)):
                last_index = last_index-1
                break
            else:
                last_index -= 1

        line_parse_start = ""
        line_parse_end = ""

        #Checking if the line before the current block can also be parsed after tokeninizing by spaces. 
        if(start_index != 0):
            ind = start_index-1
            line = example_list[ind]
            try:
                tokenized_sentence = line.strip("```").split(" ")
            except:
                tokenized_sentence = line.strip("```")
            line_parse_start = extract_parsable_code(start = True, code = code, code_list = tokenized_sentence, top = 0, bottom = len(tokenized_sentence), pl = pl)
       
        flag = 1
       
        if line_parse_start == None:
            line_parse_start = ""
        end_split = line_parse_start.strip().split(" ")
       
        if(len(end_split) == 1):
            if(end_split[0].isalnum()):
                flag = 0
        if(check_syntax(line_parse_start+"\n"+code,pl) and flag):
            code = line_parse_start + "\n"+ code

        #Checking if the line after the current block can also be parsed after tokeninizing by spaces. 
        if(last_index != (len(example_list)-1)):
            ind = last_index+1
            line = example_list[ind]
            try:
                tokenized_sentence = line.strip("```").split(" ")
            except:
                tokenized_sentence = line.strip("```")
            line_parse_end=extract_parsable_code(start = False, code = code, code_list = tokenized_sentence, top = 0, bottom = len(tokenized_sentence), pl = pl)
        
        if line_parse_end == None:
            line_parse_end = ""
        end_split = line_parse_end.strip().split(" ")
        
        if(len(end_split) == 1):
            if(end_split[0].isalnum()):
                code = remove_blank_lines(code.strip())
                return [0,code]
        
        if(check_syntax(code + "\n" + line_parse_end, pl)):
            code+="\n" + line_parse_end
        
        code = remove_blank_lines(code.strip())
       
        return [0,code]