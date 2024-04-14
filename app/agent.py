from func import *
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import langchain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.redis import Redis
from langchain_openai import OpenAI
from redisvl.extensions.llmcache import SemanticCache
from redisvl.utils.vectorize import OpenAITextVectorizer
from redisvl.utils.vectorize import HFTextVectorizer
from nemoguardrails import LLMRails, RailsConfig
import asyncio
import nest_asyncio

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['OPENAI_API_KEY'] = "Enter API KEY here"
api_key  = "Enter API KEY here"

def app_setup():
    global embedding 
    embedding= OpenAIEmbeddings(api_key=api_key)
    global text_splitter
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 150
    )
    global vectordb
    vectordb = Redis.from_texts(
    texts=[''],
    embedding = embedding,
    redis_url="redis://localhost:6379/"
    )
    
    hf = HFTextVectorizer(model="sentence-transformers/all-mpnet-base-v2")
    global semantic_cache
    semantic_cache = SemanticCache(
    name="ntest",                    
    prefix="ntest",                   
    redis_url="redis://localhost:6379",
    distance_threshold=0.1,
    vectorizer=hf
    )
    
    start_llm()
    guardrails()
    print("setup complete")
    return 


def use_pdf(file_name: str):
    text = load_pdf(file_name)
    add_to_vectordb(text)
    return 


def add_to_vectordb(text):
    splits = text_splitter.split_text(text)  
    vectordb.add_texts(splits)
    return

def start_llm():
    global llm
    llm = OpenAI(openai_api_key=api_key)
    global mretriever
    mretriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"distance_threshold": 0.8})
    template = """Use the following pieces of context given inside ``` to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}
    Question: {question}
    Helpful Answer:"""
   
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    global qa_chain 
    qa_chain= RetrievalQA.from_chain_type(
        llm,
        retriever=mretriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    
    # qa_chain = RetrievalQA.from_chain_type(
    # llm=rails.llm, chain_type="stuff", retriever=mretriever)
    return

def llm_call(inputs: str):
    print("llm call is here: ",inputs)
    result = qa_chain({"query": inputs})
    print("llm call is done: ",result)
    return result['result']

def guardrails():
    config = RailsConfig.from_path("config")
    global rails
    rails = LLMRails(config)
    rails.register_action(llm_call,name="llm_call")
    return

def check_cache(ques: str):
    if answer := semantic_cache.check(prompt=ques,return_fields=["prompt", "response", "metadata"],):
        return answer[0]
    return 0
    
def set_cache(ques,result):
    semantic_cache.store(
        prompt=ques,
        response=result,
    )



def chat_llm(ques: str):
    nest_asyncio.apply()
    print("Start is here")
    result = check_cache(ques)
    print("After cache check")
    result==0
    if result==0:
        # result = qa_chain({"query": ques})
        #result = input("answer: ")
        result = rails.generate(prompt=ques)
        print("after rail: ",result)
        set_cache(ques,result)
        return result
    print("End is here: ",result['response'])
    return result['response']