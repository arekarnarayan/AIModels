from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

#os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

## Langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

## Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user","Question:{question}")
    ]
)

## streamlit framework
st.title('Langchain Demo With LLAMA2 API')
input_text=st.text_input("Search the topic u want")

# openAI LLm 
#llm=ChatOpenAI(model="gpt-3ls.5-turbo", temparature=0.5, max_tokens=256)
#llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.5, max_tokens=256)

# ollama LLAma2 LLm 
llm=Ollama(model="llama3.2")  # Changed to "llama2"
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))
