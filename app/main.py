from fastapi import FastAPI
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import File, UploadFile
import json
import shutil
import nest_asyncio
import asyncio
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
import os
from agent import ragbot


app = FastAPI()
templates = Jinja2Templates(directory="templates")
bot = ragbot()


# This function returns the frontend of chatbot
@app.get("/", response_class=HTMLResponse)
async def chat_index(request: Request):
    return templates.TemplateResponse(
        request=request, name="chat.html", context={"id": ""}
    )

# This function takes in user query and returns the response
@app.post("/chat")
async def process_chat_input(request: Request):
    # Process the user input
    body = await request.body()
    user_message = json.loads(body.decode("utf-8"))
    loop = asyncio.get_event_loop()
    response_message = bot.chat_llm(user_message["user_message"])
    return {"response_message": response_message}


UPLOAD_DIR = "uploads"  
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# This function is used to store documents
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    print(file.filename)
    global file_path
    # Save the file to the specified directory
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    ready = bot.use_pdf(file_path)
    return 

       
