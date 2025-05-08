from langchain.memory import ConversationBufferWindowMemory
from langchain_community.llms import llamacpp
from langchain.chains import LLMChain


llm =  llamacpp(
    model_path = "/home/rteam2/m15kh/LLM/checkpoints/Phi-3-mini-4k-instruct-fp16.gguf",
    n_gpu_layers=-1,
    max_tokens=300,
    n_ctx=4096,
    seed=42,
    verbose=False
    )



meonry = ConversationBufferWindowMemory(k = 2, chat_memory='history_chat')

chain_llm = LLMChain(llm = llm, mwmory = meonry)


