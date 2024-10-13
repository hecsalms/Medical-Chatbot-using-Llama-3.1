from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from pinecone_text.sparse import BM25Encoder # uses TF-DF technique by default
from langchain_community.retrievers import PineconeHybridSearchRetriever

# Extract data from PDF:

def load_pdf(data):
    loader = DirectoryLoader(data,
                    glob = "*.pdf",
                    loader_cls = PyPDFLoader)
    
    documents = loader.load()

    return documents

# Create text chunks:

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks


# Load embedding model:

def obtain_hugging_face_embedding(HuggingFaceKey):
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key = HuggingFaceKey,
                                                   model_name = "sentence-transformers/all-MiniLM-L6-v2")

    return embeddings


""" # Get the sparse vector generator:
def get_sparse_vector_generator(text_chunks):
    # Define the sparse vector generator:
    bm25_encoder = BM25Encoder().default()

    # TF-DF values on the data:
    bm25_encoder.fit([t.page_content for t in text_chunks])

    # Store the values to a json file:
    bm25_encoder.dump("bm25_values.json")

    # Load to your BM25Encoder object:
    bm25_encoder = BM25Encoder().load("bm25_values.json")

    return bm25_encoder

# Get the retriever:
def get_retriever(embeddings, bm25_encoder, index):
    retriever = PineconeHybridSearchRetriever(embeddings = embeddings, sparse_encoder = bm25_encoder, index = index)

    return retriever """
