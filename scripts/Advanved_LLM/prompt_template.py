from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from pprint import pprint

llm = LlamaCpp(
model_path="/home/rteam2/m15kh/LLM/checkpoints/Phi-3-mini-4k-instruct-fp16.gguf",
n_gpu_layers=-1,
max_tokens=500,
n_ctx=2048,
seed=42,
verbose=False
)


template = """
<s><|user|>{input_prompt}<|end|><|assistent|><|end|>
"""

prompt = PromptTemplate(
    template=template,
    input_variables = ["input_prompt"]
)

basic_chain = prompt | llm

ans = basic_chain.invoke("hi my name is mohammad what 's sun of 2+54")
print(ans)