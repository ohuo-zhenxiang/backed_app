import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi_pagination import add_pagination
from loguru import logger
from starlette.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api.router import api_router
from scheduler_utils import Scheduler
from settings import LOGGING_DIR, APP_PORT, APP_HOST

logger.add(f"{LOGGING_DIR}/main_server_log.log", rotation="500MB",
           encoding="utf-8", enqueue=True, retention="30 days",
           format="{time:YY-MM-DD HH:mm:ss} | {extra[name]} | {level} | {message}")
logger = logger.bind(name="MainServer")


@asynccontextmanager
async def lifespan(app: FastAPI):
    Scheduler.start()
    yield
    Scheduler.shutdown()

app = FastAPI(title="人脸服务管理系统",
              openapi_url="/api/openapi.json",
              version="v0.0.1",
              description="用于人脸服务的后台管理系统，支持多进程定时后台任务处理、实时消息推送",
              lifespan=lifespan)

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_methods=["*"],
                   allow_headers=["*"])

app.mount("/FaceImageData", StaticFiles(directory="FaceImageData"), name="FaceImageData")
app.mount("/TaskRecord", StaticFiles(directory="TaskRecord"), name="TaskRecord")

add_pagination(app)
app.include_router(api_router, prefix='/api')


@app.get("/", include_in_schema=False)
async def redict_to_docs():
    return RedirectResponse(url="/docs")


# 这种写法弃用了
# def register_init(app: FastAPI) -> None:
#     @app.on_event("startup")
#     async def init_connect():
#         # 初始化apscheduler
#         Scheduler.start()
#
#     @app.on_event("shutdown")
#     async def shutdown_connect():
#         # 关闭apscheduler
#         Scheduler.shutdown()
#
#
# register_init(app)


if __name__ == "__main__":
    import multiprocessing
    from datetime import datetime
    from initial_data import run_init

    multiprocessing.freeze_support()

    run_init()
    print('----初始化数据库完成----')

    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.success(f"Linking Start")

    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
