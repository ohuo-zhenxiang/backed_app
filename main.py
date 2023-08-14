from fastapi import FastAPI
import uvicorn
from starlette.middleware.cors import CORSMiddleware
from api.router import api_router
from fastapi_pagination import add_pagination
from pprint import pprint

app = FastAPI(title="Project_dev", openapi_url="/api/openapi.json", version="0.0.0", description="fastapi")

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_methods=["*"],
                   allow_headers=["*"])

add_pagination(app)
app.include_router(api_router, prefix='/api')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9527)
