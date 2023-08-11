from fastapi import FastAPI, Request
import uvicorn
from starlette.middleware.cors import CORSMiddleware
from pprint import pprint
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_methods=["*"],
                   allow_headers=["*"])


class Post_data(BaseModel):
    username: str
    password: str


@app.post("/api/login/access-token")
def get_what(postData:Post_data):
    data = postData.json()
    pprint(data)


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=9527)
