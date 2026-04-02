import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Configuration: Load environment variables from .env
load_dotenv()

# 2. Initialization: Connect to the Pinecone vector database
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "sentry-index"

# 3. Ingestion: Load PDF files from the local 'docs' directory
loader = PyPDFDirectoryLoader("./docs")
docs = loader.load()

# 4. Processing: Split documents into chunks for retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 5. Vectorization: Generate 1536-dimension embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeVectorStore.from_documents(
    splits, embeddings, index_name=index_name
)

# 6. Pipeline: Define the RAG logic using LCEL (Modern Syntax)
llm = ChatOpenAI(model="gpt-4o", temperature=0)
retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# This chain bypasses the 'langchain.chains' module entirely
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. Execution: Execute query and output result
if __name__ == "__main__":
    #query = "What are the key technical requirements mentioned in this document?"
    query = "Summarize the main purpose of this document."

    # Can change query to test different results
    response = rag_chain.invoke(query)

    print("\n--- SentryQuery Result ---")
    print(response)