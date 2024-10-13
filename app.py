from flask import Flask, render_template, jsonify, request
from src.helper import load_pdf, text_split, obtain_hugging_face_embedding
# from store_index import retriever
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pinecone_text.sparse import BM25Encoder # uses TF-DF technique by default
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from src.prompt import *
import os
# import nltk

app = Flask(__name__)

# Take the enviroment variables from .env directory:
load_dotenv() 

GroqKey = os.getenv('GC_KEY')
PineconeKey = os.getenv('PN_KEY')
HuggingFaceKey = os.getenv('HF_KEY')

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
# The following method does not implies model downloading:
# Define the dense vector generator:
embeddings = obtain_hugging_face_embedding(HuggingFaceKey)

# Choose one of the following names to avoid code repetition:

index_name = "semantic-search-langchain-pinecone"

# Initialize the Pinecone client:

pc = Pinecone(api_key = os.environ.get('PN_KEY'))

# Create the index:

if index_name not in pc.list_indexes().names():
  pc.create_index(
      name = index_name,
      dimension = 384, # dimension of dense vector
      metric = "dotproduct",
      spec = ServerlessSpec(cloud = "aws", region = "us-east-1")
  )

# Load the index:
index = pc.Index(index_name)

# Alternative, but maybe deprecated way:
# retrieval = Pinecone.from_existing_index(index_name, embeddings)

# Define the prompt:
PROMPT = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])

# Format it as dictionary:
chain_type_kwargs = {"prompt": PROMPT}

# Instantiate the model class:
LLM = ChatGroq(
    temperature = 0,
    groq_api_key = GroqKey,
    model_name = "llama-3.1-70b-versatile"
)

# nltk.download('punkt_tab')

# Define the sparse vector generator:

bm25_encoder = BM25Encoder().default()

# TF-DF values on the data:
bm25_encoder.fit([t.page_content for t in text_chunks])

# Store the values to a json file:
bm25_encoder.dump("bm25_values.json")

# Load to your BM25Encoder object:
bm25_encoder = BM25Encoder().load("bm25_values.json")

# Get the retriever object to search for similar embeddings:
retriever = PineconeHybridSearchRetriever(embeddings = embeddings, sparse_encoder = bm25_encoder, index = index)

qa = RetrievalQA.from_chain_type(llm = LLM, 
                                  chain_type = "stuff", 
                                  retriever = retriever,
                                  return_source_documents = True,
                                  chain_type_kwargs = chain_type_kwargs)

@app.route("/")
def index():
  return render_template('chat.html')

@app.route("/get", methods = ["GET", "POST"])
def chat():
  msg = request.form["msg"]
  input = msg
  print(input)
  result = qa({"query": input})
  print("Response : ", result["result"])
  return str(result["result"])

if __name__ == '__main__':
  app.run(host = "0.0.0.0", port = 8080, debug = True)