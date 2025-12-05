#!/usr/bin/env python
# coding: utf-8

# In[20]:


from langchain.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.schema import Document
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
import os


# In[113]:


load_dotenv()


# In[63]:


HF_KEY = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")


# In[22]:


#1. Load PDF document
loader = PyPDFLoader('/Users/anushkabansal/Desktop/RAG Chatbot Project/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf')
docs = loader.load()


# In[23]:


#2. Split text into smaller chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=50,
)
chunks = splitter.split_documents(docs)


# In[25]:


#3. Create embeddings and store in Chroma vector database
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    embedding_function=embedder,
    persist_directory='my_chroma_db_files1',
    collection_name='sample'
)

vector_store.add_documents(chunks)


# In[26]:


#4. Setup retriever with Multiquery

retriever = vector_store.as_retriever(
    search_type="mmr",                   # <-- This enables MMR
    search_kwargs={"k": 3, "lambda_mult": 0.75}  # k = top results, lambda_mult = relevance-diversity balance
)


# In[ ]:


#5. Creating a promt template

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)


# In[96]:


question="what is cancer?"
retrieved_docs    = retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
context_text


# In[97]:


final_prompt = prompt.invoke({"context": context_text, "question": question})


# In[98]:


type(final_prompt)


# In[114]:


#5. Setup LLM

from langchain_groq import ChatGroq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
    max_tokens=500
)


# In[116]:


answer = llm.invoke("what is cancer")
print(answer.content)


# In[117]:


from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# In[118]:


def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text


# In[119]:


parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})


# In[120]:


parser = StrOutputParser()


# In[121]:


main_chain = parallel_chain | prompt | llm | parser


# In[125]:


main_chain.invoke('what are the side effects of caffieine?')


# In[ ]:




