from typing import List

from fastapi import FastAPI, Request, Form, File, UploadFile
import uvicorn
from starlette.middleware.cors import CORSMiddleware
from pprint import pprint
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_methods=["*"],
                   allow_headers=["*"])



@app.post("/api/faces/add_test_face")
async def get_what(file: UploadFile = File()):
    print(file)
    return {"result": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=9527)
