from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Llamacpp
from SmartAITool.core import bprint
from pprint import pprint
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser

llm = llamacpp(
    
model_path="/home/rteam2/m15kh/LLM/checkpoints/Phi-3-mini-4k-instruct-fp16.gguf",
n_gpu_layers=-1,
max_tokens=500,
n_ctx=4096,
seed=42,
verbose=False
)


template = """<|user|>
current conversation:{chat_history}
{input_prompt}
<|end|>
<|assistant|>"""

prompt = PromptTemplate(template=template,
                        input_variables=['input_prompt', 'chat_history']
                        
                        )

memory = ConversationBufferMemory(memory_key='chat_history')


llm_chain = Runnable(
    lambda input :{
        "chat_history":memory.chat_memory.messages,
        "input": input["input"]
        
    }
) | prompt | llm | StrOutputParser



result = llm_chain.invoke({"input_prompt": "Hi! My name is Maarten. What is 1 + 1?"})
bprint()
pprint(result)
result2 = llm_chain.invoke({"input_prompt": "What is my name?"})
bprint()
pprint(result2)