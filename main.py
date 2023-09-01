import atexit

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
from starlette.middleware.cors import CORSMiddleware
from scheduler_utils import Scheduler

from fastapi_pagination import add_pagination
from logger_module import Logger
from pprint import pprint
from api.router import api_router

from logger_module import logger

app = FastAPI(title="Project_dev", openapi_url="/api/openapi.json", version="0.0.0", description="fastapi")

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_methods=["*"],
                   allow_headers=["*"])

app.mount("/FaceImageData", StaticFiles(directory="FaceImageData"), name="FaceImageData")
app.mount("/TaskRecord", StaticFiles(directory="TaskRecord"), name="TaskRecord")

add_pagination(app)
app.include_router(api_router, prefix='/api')


def register_init(app: FastAPI) -> None:
    @app.on_event("startup")
    async def init_connect():
        # 初始化apscheduler
        Scheduler.start()

    @app.on_event("shutdown")
    async def shutdown_connect():
        # 关闭apscheduler
        Scheduler.shutdown()


register_init(app)

if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    Logger.success("System Start!!!")

    uvicorn.run(app, host="0.0.0.0", port=9527)
