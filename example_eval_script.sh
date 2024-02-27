python3 src/evaluation.py --model_path Phind/Phind-CodeLlama-34B-v2 --data_subset maintainability --metric diffbleu --score_k 1 && 
python3 src/evaluation.py --model_path Phind/Phind-CodeLlama-34B-v2 --data_subset maintainability --metric codeql --score_k 1 && 
python3 src/evaluation.py --model_path Phind/Phind-CodeLlama-34B-v2 --data_subset maintainability --metric codeql-diffbleu --score_k 1