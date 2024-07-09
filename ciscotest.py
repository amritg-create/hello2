# PYTHON FOR GENERATIVE AI COURSE
# RETRIEVAL-AUGMENTED GENERATION (RAG)
# ***

# CHALLENGE 1: CREATE A DATA SCIENCE EXPERT USING THE INTRODUCTION TO STATISTICAL LEARNING WITH PYTHON PDF

# DIFFICULTY: BEGINNER

# streamlit run path_to_app.py

"""
What are the top 3 things needed to do principal component analysis (pca)?
"""

from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

import streamlit as st
import yaml
import uuid



from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import pandas as pd
import yaml
from pprint import pprint


OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# PDF Loader 

loader = PyPDFLoader('cisco_pdfs\Q3 Cisco Report.pdf')

# THIS TAKES 5 MINUTES...
documents = loader.load()

CHUNK_SIZE = 1000

text_splitter = CharacterTextSplitter(
    chunk_size=CHUNK_SIZE, 
    # chunk_overlap=100,
    separator="\n"
)

docs = text_splitter.split_documents(documents)

docs

len(docs)

docs[0]

pprint(dict(docs[5])["page_content"])

docs[5]

# Vector Database

embedding_function = OpenAIEmbeddings(
    model='text-embedding-ada-002',
    api_key=OPENAI_API_KEY
)

vectorstore = Chroma.from_documents(
    docs, 
    persist_directory="Cisco_Q3summarizer/data/chroma_cisco_learning.db",
    embedding=embedding_function
)


vectorstore = Chroma(
    persist_directory="Cisco_Q3summarizer/data/chroma_cisco_learning.db",
    embedding_function=embedding_function
)

retriever = vectorstore.as_retriever()

retriever

# RAG LLM Model

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(
    model = 'gpt-3.5-turbo',
    temperature = 0.7,
    api_key=OPENAI_API_KEY
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

result = rag_chain.invoke("What was Cisco total revenues in third quarter?")

pprint(result)


result = rag_chain.invoke("What was Cisco product revenues in third quarter?")

pprint(result)



# Initialize the Streamlit app
st.set_page_config(page_title="Your Cisco Q3 Earnings Statement Copilot", layout="wide")
st.title("Your Cisco Earnings Copilot")

# Load the API Key securely
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Set up Chat Memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

view_messages = st.expander("View the message contents in session state")

def create_rag_chain(api_key):
    
    embedding_function = OpenAIEmbeddings(
        model='text-embedding-ada-002',
        api_key=api_key,
        chunk_size=500,
    )
    vectorstore = Chroma(
        persist_directory= "Cisco_Q3summarizer/data/chroma_cisco_learning.db",
        embedding_function=embedding_function
    )
    
    retriever = vectorstore.as_retriever()
    
    llm = ChatOpenAI(
        model="gpt-3.5-turbo", 
        temperature=0.4, 
        api_key=api_key,
        max_tokens=4000,
    )

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # * 2. Answer question based on Chat Context
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # * Combine both RAG + Chat Message History
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: msgs,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

rag_chain = create_rag_chain(OPENAI_API_KEY)

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if question := st.chat_input("Enter your question here:", key="query_input"):
    with st.spinner("Thinking..."):
        st.chat_message("human").write(question)     
           
        response = rag_chain.invoke(
            {"input": question}, 
            config={
                "configurable": {"session_id": "any"}
            },
        )
        # Debug response
        # print(response)
        # print("\n")
  
        st.chat_message("ai").write(response['answer'])

# * NEW: View the messages for debugging
# Draw the messages at the end, so newly generated ones show up immediately
with view_messages:
    """
    Message History initialized with:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    ```

    Contents of `st.session_state.langchain_messages`:
    """
    view_messages.json(st.session_state.langchain_messages)
