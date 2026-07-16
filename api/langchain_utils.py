from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from typing import List
from langchain_core.documents import Document
import os
from .chroma_utils import vectorstore

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

output_parser = StrOutputParser()


def get_llm(model=None):
    selected_model = model or os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
    return ChatGoogleGenerativeAI(model=selected_model, temperature=0)


# Set up prompts and chains
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])



document_prompt = ChatPromptTemplate.from_template(
    "Source: {source}\nContent: {page_content}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an assistant that answers questions strictly using the provided context. "
     "Each piece of context is labeled with its Source filename. "
     "Only use information found in the context below to answer the user's question. "
     "If the answer is not contained in the context, say clearly: "
     "\"I don't have enough information in the uploaded documents to answer that.\" "
     "Do not use any outside knowledge, even if you know the answer. "
     "At the end of your answer, cite which source file(s) you used, like: (Source: filename.pdf)"
    ),
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])


def get_rag_chain(model="gemini-3.1-flash-lite"):
    llm = get_llm(model)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(
        llm,
        qa_prompt,
        document_prompt=document_prompt
    )
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain