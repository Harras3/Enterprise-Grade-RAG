from fastapi import FastAPI
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import File, UploadFile
import json
import shutil
import nest_asyncio
import os
from agent import *


app = FastAPI()
templates = Jinja2Templates(directory="templates")
app_setup()
print("Setup Complete")


@app.get("/", response_class=HTMLResponse)
async def chat_index(request: Request):
    
    return templates.TemplateResponse(
        request=request, name="chat.html", context={"id": ""}
    )


@app.post("/chat")
async def process_chat_input(request: Request):
    # Process the user input
    body = await request.body()
    user_message = json.loads(body.decode("utf-8"))
    loop = asyncio.get_event_loop()
    response_message = chat_llm(user_message["user_message"])
    return {"response_message": response_message}


UPLOAD_DIR = "uploads"  
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    print(file.filename)
    global file_path
    # Save the file to the specified directory
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    ready = use_pdf(file_path)
    return

       
