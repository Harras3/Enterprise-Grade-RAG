from fastapi import FastAPI

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import File, UploadFile
import json
import shutil
import nest_asyncio


# from agent import *

file_name = "data.pdf"
app = FastAPI()
# ready = start_pdf(file_name)
# templates = Jinja2Templates(directory="templates")



# @app.get("/", response_class=HTMLResponse)
# async def chat_index(request: Request):
#     app_setup()
#     return templates.TemplateResponse(
#         request=request, name="chat.html", context={"id": ""}
#     )


# @app.post("/chat")
# async def process_chat_input(request: Request):
#     # Process the user input
#     body = await request.body()
#     user_message = json.loads(body.decode("utf-8"))
#     loop = asyncio.get_event_loop()
#     # response_message = await loop.run_in_executor(None, process_input(user_message["user_message"]))
#     response_message = chat_llm(user_message["user_message"])
#     return {"response_message": response_message}


# # if __name__ == "__main__":
# #     import uvicorn

# #     uvicorn.run(app, host="127.0.0.1", port=8000)
# UPLOAD_DIR = "uploads"  
# if not os.path.exists(UPLOAD_DIR):
#     os.makedirs(UPLOAD_DIR)

# @app.post("/upload")
# async def upload_file(file: UploadFile = File(...)):
#     print(file.filename)
#     global file_path
#     # Save the file to the specified directory
#     file_path = os.path.join(UPLOAD_DIR, file.filename)
#     with open(file_path, "wb") as f:
#         shutil.copyfileobj(file.file, f)
#     ready = use_pdf(file_path)
#     return

       
# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="127.0.0.1", port=8000)