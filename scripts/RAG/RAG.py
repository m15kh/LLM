import yaml
import os
from pprint import pprint
from SmartAITool.core import bprint

# Load API configuration from a YAML file
with open("/home/rteam2/m15kh/LLM/config.yaml", "r") as file:
    config = yaml.safe_load(file)

os.environ["LANGSMITH_TRACING"] = config.get("LANGSMITH_TRACING", "false")
os.environ["LANGSMITH_API_KEY"] = config["LANGSMITH_API_KEY"]
os.environ["TAVILY_API_KEY"] = config["TAVILY_API_KEY"]
os.environ['HUGGINGFACEHUB_API_TOKEN'] = config['HUGGINGFACEHUB_API_TOKEN']
GPT_API_KEY = config['GPT_API_KEY']

from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model

llm = init_chat_model(
    api_key=GPT_API_KEY,
    base_url="https://api.gilas.io/v1/",
    model="gpt-4o-mini"
)

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    api_key=GPT_API_KEY,
    base_url="https://api.gilas.io/v1/",
    model="text-embedding-3-large",
    dimensions=3072
)

from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)
print((vector_store))

import bs4
from langchain_community.document_loaders import WebBaseLoader

# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

assert len(docs) == 1
print(f"Total characters: {len(docs[0].page_content)}")

from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

# Token encoding to measure chunk token sizes
encoding = tiktoken.get_encoding("cl100k_base")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,  # safer chunk size to avoid hitting 8191 token limit
    chunk_overlap=500,
    add_start_index=True,
)

all_splits = text_splitter.split_documents(docs)

# Print token count for each split (for debugging)
for i, split in enumerate(all_splits):
    num_tokens = len(encoding.encode(split.page_content))
    print(f"Split {i}: {num_tokens} tokens")

print(f"Split blog post into {len(all_splits)} sub-documents.")

# Safely embed in batches of 1 to avoid exceeding total token limit per request
document_ids = []
for split in all_splits:
    ids = vector_store.add_documents(documents=[split])
    document_ids.extend(ids)

print(document_ids[:3])
