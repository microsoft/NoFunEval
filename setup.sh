#!/bin/bash
pip3 install -r requirements.txt

mkdir codeql-home 
wget https://github.com/github/codeql-cli-binaries/releases/download/v2.5.0/codeql.zip -P codeql-home/
unzip codeql-home/codeql.zip -d codeql-home/
 
git clone https://github.com/github/codeql.git codeql-home/codeql-repo
cd codeql-home/codeql-repo
git checkout 20416ae0342c66aa05bc099af8e5a020b018a978
  
codeql-home/codeql/codeql resolve languages
codeql-home/codeql/codeql resolve qlpacks

cd ../../

mv  -v src/evaluation/qls_for_security/cpp/* codeql-home/codeql-repo/cpp/ql/src/
mv  -v src/evaluation/qls_for_security/python/* codeql-home/codeql-repo/python/ql/src/

cd src/evaluation/tree-sitter/

git clone https://github.com/tree-sitter/tree-sitter-cpp.git 
git clone https://github.com/tree-sitter/tree-sitter-c.git 
git clone https://github.com/tree-sitter/tree-sitter-python.git 
git clone https://github.com/tree-sitter/tree-sitter-java.git 
git clone https://github.com/tree-sitter/tree-sitter-javascript.git 
git clone https://github.com/tree-sitter/tree-sitter-scala.git 
git clone https://github.com/jiyee/tree-sitter-objc.git 
git clone https://github.com/fwcd/tree-sitter-kotlin.git