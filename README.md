
![Logo](https://github.com/Harras3/Production-Grade-RAG/blob/main/img/logo.jpg?raw=true)

# Enterprise Grade RAG Solution Template

This is a template for Enterprise Grade RAG solution focusing on Low latency. The code is very modular and contains the most important elements and can be very easily extended for your needs.






## Elements of RAG

- Redis based Vector DB
When a user uploads a document, it is chunked and passed through a embedding model, both chunk and embedding are stored in Redis.

- Redis based Semantic cache
A cache is used to store query embedding against previous chatbot responses. This is very important element because if your RAG is based upon some specific data most of the questions will be very similar so inorder to avoid redundant calls to a LLM.
- Nemo Guardrails
Guardrails can be used before call to LLM to check for off topic quries, this is also very important as it will not allow the quries which are not related to the topic for which the RAG is being designed and will save the cost of LLM call.

## Flow of RAG
![App Screenshot](https://github.com/Harras3/Production-Grade-RAG/blob/main/img/flow.jpg?raw=true)

1) When a user enter a query on HTML frontend, it is passed to FAST API server.
2) The user query will be passed to semantic cache and will be checked if exists in cache. If it exists then the response stored will be returned to the user.
3) If query is not present in cache then the query will be passed to Nemo Guardrails.
4) Guardrails will check the query and will follow that predefined flow which is semanticallly near to the query.
5) I have defined a flow in guardrails that if query doesn't take any other flow it should be directed to a function which passes the query to Langchain QA chain.
6) The chain will pass the query to LLM and the retrieved response will be saved in semantic cache and will be returned to the user on HTML page.







# Deployment

Firstly redis server should be setup. We will use docker for it.
Run the command below after downloading docker.
```bash
  docker run -d --name redis-stack-server -p 6379:6379 redis/redis-stack-server:latest
```
After this clone this repo and in the app folder
run the command
```bash
   pip install -r requirements.txt
```
As Langchain and vectordb is using gpt 3.5 and embedding model so api-key is needed. You need to write your own api key inside the agent.py file.

Finally run the following command to setup a run FAST API server.

```bash
   python -m uvicorn main:app --reload
```
## Contributers

https://github.com/salmanjann

Thank you for developing the frontend.


