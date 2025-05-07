from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from SmartAITool.core import bprint , cprint
from pprint import pprint
from langchain_core.runnables import RunnableLambda


llm = LlamaCpp(
model_path="/home/rteam2/m15kh/LLM/checkpoints/Phi-3-mini-4k-instruct-fp16.gguf",
n_gpu_layers=-1,
max_tokens=500,
n_ctx=4096,
seed=42,
verbose=False
)



title_template = """
<|user|>create a title for a story {summary}. only return the title.
<|end|>
<|assistant|>
"""

title_prompt = PromptTemplate(template = title_template, input_variables = ["summary"])

title_chain = title_prompt | llm 

# summary_input = {"summary": "a girl that lost her mother"}
# title_result  = title_chain.invoke(summary_input)
# cprint("Generated Title:\n", 'blue')
# pprint(title_result)
# bprint()

#-----------------------------------------------------------------------------
character_template = """<|user|>
Describe the main character of a story about {summary} with the title {title}.
Use only two sentences.<|end|>
<|assistant|>"""


character_prompt = PromptTemplate(
template=character_template, input_variables=["summary", "title"]
)

character_chain =  character_prompt | llm


# character_result = character_chain.invoke({
#     "summary": summary_input['summary'],
#     "title": title_result
    
# })
# cprint("\nCharacter Description:\n", 'blue')
# pprint(character_result)


#---------------------------------------------------------------------------

story_template = """
<|user|>Create a story about {summary} with the title {title}. The main character is:
{character}. Only return the story and it cannot be longer than one paragraph.<|end|>
<|assistant|><|end|>
"""
story_prompt = PromptTemplate(template=story_template, input_variables=["summary","title", "character"])
story_chain = story_prompt | llm

# story_result = story_chain.invoke({
#     "summary": summary_input['summary'],
#     "title": title_result,
#     "character": character_result
# })



def chain_func(inputs):
    summary = inputs["summary"]
    title = title_chain.invoke({"summary": summary})
    character = character_chain.invoke({"summary": summary, "title": title})
    story = story_chain.invoke({"summary": summary, "title": title, "character": character})
    return story

llm_chain = RunnableLambda(chain_func)

summary_input = {"summary": "a girl that lost her mother"}
llm_result = llm_chain.invoke(summary_input)
cprint('story')
pprint(llm_result)
    