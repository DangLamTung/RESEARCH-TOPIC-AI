# 3.1. Import libraries
import streamlit as st
from llama_index.core import PromptTemplate, VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from IPython.display import Markdown, display
import chromadb

chromadb.api.client.SharedSystemClient.clear_system_cache()
import os

from llama_index.llms.gemini import Gemini
GOOGLE_API_KEY = ""  # add your GOOGLE API key here
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY



llm = Gemini(
    model="models/gemini-2.0-flash",
    # api_key="some key",  # uses GOOGLE_API_KEY env var by default
)

# sentence transformers
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="keepitreal/vietnamese-sbert")

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
    "Imagine you are a insurance's assistant and "
    "you answer a recruiter's questions about the  insurance's policy."
    "Here is some context from the insurance's "
    "resume related to the query::\n"
    "-----------------------------------------\n"
    "{context_str}\n"
    "-----------------------------------------\n"
    "Considering the above information, "
    "Please respond to the following inquiry:\n\n"
    "Question: {query_str}\n\n"
    "Answer succinctly and ensure your response is "
    "truth, based on the fact stated in the context."
 
)
qa_template = PromptTemplate(template)

# Query Data
query_engine = index.as_query_engine(llm =llm  )
# response = query_engine.query("can llama 2 calculate 1 +1  ")

# Initialize message history

st.header("Insurance Chatbot")

if "messages" not in st.session_state.keys():  # Initialize the chat message history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi, I'm Insurance Support Chatbot!",
        }
    ]


# # 3.3. Load and index data
# @st.cache_resource(show_spinner=False)  # Cache the data loading
# def load_data():
#     with st.spinner(text="Loading and indexing your data, keep it cool..."):
#         reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
#         docs = reader.load_data()
#         service_context = ServiceContext.from_defaults(
#             llm=OpenAI(
#                 model="gpt-3.5-turbo",
#                 temperature=0.5,
#                 system_prompt="You are an expert in Cambium software company. Keep your answers informative and polite",
#             )
#         )
#         index = VectorStoreIndex.from_documents(docs, service_context=service_context)
#         return index


# index = load_data()

# 3.4. Create the chat engine
# chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# 3.5. Prompt for user input and display message history
if prompt := st.chat_input("Hỏi về sản phẩm PVI insurance"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 3.6. Pass query to chat engine and display response
# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking about your question..."):
            response = query_engine.query(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)  # Add response to message history
