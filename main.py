import atexit
import time
from settings import LOGGING_DIR
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
from starlette.middleware.cors import CORSMiddleware
from scheduler_utils import Scheduler

from fastapi_pagination import add_pagination
from loguru import logger
from pprint import pprint
from api.router import api_router

logger.add(f"{LOGGING_DIR}/main_server_log.log", rotation="500MB",
           encoding="utf-8", enqueue=True, retention="30 days",
           format="{time:YY-MM-DD HH:mm:ss} | {level} | {extra[name]} | {message}")
logger = logger.bind(name="MainServer")

app = FastAPI(title="人脸服务管理系统", openapi_url="/api/openapi.json", version="v0.0.0",
              description="用于人脸服务的后台管理系统，支持多进程定时后台任务处理、实时消息推送")

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
    from datetime import datetime

    multiprocessing.freeze_support()
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.success(f"Linking Start")

    uvicorn.run(app, host="0.0.0.0", port=9527)
