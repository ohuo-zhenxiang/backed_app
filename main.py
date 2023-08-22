from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
from starlette.middleware.cors import CORSMiddleware
from api.router import api_router
from fastapi_pagination import add_pagination
from pprint import pprint
import atexit
from pytz import timezone

Scheduler = BackgroundScheduler(timezone=timezone("Asia/Shanghai"))

app = FastAPI(title="Project_dev", openapi_url="/api/openapi.json", version="0.0.0", description="fastapi")

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_methods=["*"],
                   allow_headers=["*"])

app.mount("FaceImageData", StaticFiles(directory="FaceImageData"), name="FaceImageData")

add_pagination(app)
app.include_router(api_router, prefix='/api')

# 停止服务时也关闭调度器
atexit.register(Scheduler.shutdown())

if __name__ == "__main__":
    # 开启调度器
    Scheduler.start()
    uvicorn.run(app, host="0.0.0.0", port=9527)
