from llama_index.core import PromptTemplate, VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from IPython.display import Markdown, display
import chromadb

import os

GOOGLE_API_KEY = ""  # add your GOOGLE API key here
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


from llama_index.llms.gemini import Gemini

llm = Gemini(
    model="models/gemini-2.0-flash",
    # api_key="some key",  # uses GOOGLE_API_KEY env var by default
)

# sentence transformers
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

# create client and a new collection
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("quickstart2")
# load documents
documents = SimpleDirectoryReader("./data/").load_data()

# set up ChromaVectorStore and load in data
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)

# print(documents)

template = (
    "Imagine you are a data scientist's assistant and "
    "you answer a recruiter's questions about the data scientist's experience."
    "Here is some context from the data scientist's "
    "resume related to the query::\n"
    "-----------------------------------------\n"
    "{context_str}\n"
    "-----------------------------------------\n"
    "Considering the above information, "
    "Please respond to the following inquiry:\n\n"
    "Question: {query_str}\n\n"
    "Answer succinctly and ensure your response is "
    "clear to someone without a data science background."
    "The data scientist's name is Bhavik Jikadara."
)
qa_template = PromptTemplate(template)

# Query Data
query_engine = index.as_query_engine(llm =llm  )
response = query_engine.query("can llama 2 calculate 1 +1  ")

print(response)