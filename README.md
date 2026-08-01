1. conda create -n nsft python=3.10
2. conda activate nsft
3. cd llama_factory
4. pip install -e ".[torch,metrics]"
5. python pipeline_llama3\pipeline_Bio.py