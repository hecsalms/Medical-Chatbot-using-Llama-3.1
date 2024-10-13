from src.helper import load_pdf, text_split, obtain_hugging_face_embedding
from pinecone_text.sparse import BM25Encoder # uses TF-DF technique by default
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
# import nltk
import os

# Take the enviroment variables from .env directory:
load_dotenv() 

GroqKey = os.getenv('GC_KEY')
PineconeKey = os.getenv('PN_KEY')
HuggingFaceKey = os.getenv('HF_KEY')

# print(PineconeKey)

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

  index = pc.Index(index_name)

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

# Get the embeddings of the documents on the vector database
retriever.add_texts([t.page_content for t in text_chunks])