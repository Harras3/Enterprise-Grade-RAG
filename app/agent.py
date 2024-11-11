from func import load_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import langchain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores.redis import Redis
from langchain_huggingface import HuggingFaceEmbeddings
from redisvl.extensions.llmcache import SemanticCache
from redisvl.utils.vectorize import OpenAITextVectorizer
from redisvl.utils.vectorize import HFTextVectorizer
# from nemoguardrails import LLMRails, RailsConfig
from langchain_cerebras import ChatCerebras
import asyncio
import nest_asyncio

os.environ["TOKENIZERS_PARALLELISM"] = "false"
c_api_key = "csk-pxf585m2cmtk5mevdmn9j522hwe82fy4r42xk4efvnff63f3"
def get_db_vectorizer():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    vectorizer = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
    return vectorizer

def get_cache_vectorizer():
    vectorizer = HFTextVectorizer(model="sentence-transformers/all-mpnet-base-v2")
    return vectorizer

def get_llm():
    llm = ChatCerebras(
        model="llama3.1-70b",api_key=c_api_key
    )
    return llm

# This class contains all the chatbot functionality
class ragbot:

    # This function is used to setup vector db and semantic cache
    def __init__(self):  
        self.embedding= get_db_vectorizer()
        self.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 150
        )
        self.vectordb = Redis.from_texts(
        texts=[''],
        embedding = self.embedding,
        redis_url="redis://localhost:6379/"
        )
        
        hf = get_cache_vectorizer()
        self.semantic_cache = SemanticCache(
        name="ntest",                    
        prefix="ntest",                   
        redis_url="redis://localhost:6379",
        distance_threshold=0.1,
        vectorizer=hf
        )
        self.start_llm()
        # self.guardrails()
        print("setup complete")
        return 


    def use_pdf(self,file_name: str):
        text = load_pdf(file_name)
        self.add_to_vectordb(text)
        return 

    # This function adds the chunks into vector db
    def add_to_vectordb(self,text):
        splits = self.text_splitter.split_text(text)  
        self.vectordb.add_texts(splits)
        return

    # This function is used to connect vectordb and llm into langchain QAchain
    def start_llm(self):
        self.llm = get_llm()
        self.mretriever = self.vectordb.as_retriever(search_type="similarity", search_kwargs={"distance_threshold": 0.8})
        template = """Use the following pieces of context given inside ``` to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        {context}
        Question: {question}
        Helpful Answer:"""
    
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template) 
        self.qa_chain= RetrievalQA.from_chain_type(
            self.llm,
            retriever=self.mretriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        return

    # This function calls langchain QAchain
    def llm_call(self,inputs: str):
        result = self.qa_chain({"query": inputs})
        return result['result']

    # This function is used to setup guardrails
    # def guardrails(self):
    #     config = RailsConfig.from_path("config")
    #     self.rails = LLMRails(config)
    #     self.rails.register_action(self.llm_call,name="llm_call")
    #     return

    # This function checks for prompt in cache
    def check_cache(self,ques: str):
        if answer := self.semantic_cache.check(prompt=ques,return_fields=["prompt", "response", "metadata"],):
            return answer[0]
        return 0
        
    # This function saves query and response into caches
    def set_cache(self,ques,result):
        self.semantic_cache.store(
            prompt=ques,
            response=result,
        )


    # The main flow of the chatbot is here
    def chat_llm(self,ques: str):
        # nest_asyncio.apply()
        result = self.check_cache(ques)
        result==0
        if result==0:
            # result = self.rails.generate(prompt=ques)
            result = self.llm_call(ques)
            self.set_cache(ques,result)
            return result
        return result['response']