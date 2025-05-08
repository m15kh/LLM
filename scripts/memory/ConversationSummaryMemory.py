from langchain.memory import ConversationSummaryMemory
from langchain_community.llms import llamacpp

llm =  llamacpp(
    model_path = "/home/rteam2/m15kh/LLM/checkpoints/Phi-3-mini-4k-instruct-fp16.gguf",
    n_gpu_layers=-1,
    max_tokens=300,
    n_ctx=4096,
    seed=42,
    verbose=False
    )

