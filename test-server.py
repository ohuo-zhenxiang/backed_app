from fastapi import FastAPI, HTTPException
from celery import Celery
from celery.schedules import crontab, timedelta
from datetime import datetime, timedelta
import uuid
import pytz

app = FastAPI()

# 配置 Celery
celery = Celery('celery_test_tasks', broker='redis://:redis@localhost:6379/4',
                backend='redis://:redis@loaclhost:6379/4')

celery.conf.timezone = 'Asia/Shanghai'
celery.conf.enable_utc = False # 关闭UTC时间



@app.on_event('startup')
def startup_event():
    celery.conf.beat_schedule = {}


@app.on_event('shutdown')
def shutdown_event():
    pass


@app.post('/tasks/create')
async def create_task(start_time: str, end_time: str, interval_minutes: int):
    """
    创建定时任务
    :param start_time: 开始时间
    :param end_time: 结束时间
    :param interval_minutes: 间隔分钟数
    :return:
    """
    try:
        task_id = str(uuid.uuid4())
        start_datetime = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')

        # 创建定时任务
        celery.conf.beat_schedule[task_id] = {
            'task': 'app.tasks.process_task',
            "schedule": timedelta(seconds='*/5', hours='16-17'),
            'args': (task_id, ),
            'options': {'expires': 10000},
        }

        return {"message": "Task created successfully", "task_id": task_id}
    except Exception as e:
        return {"error":str(e)}

@app.delete('/tasks/delete/{task_id}')
async def delete_task(task_id: str):
    try:
        del celery.conf.beat_schedule[task_id]
        return {"message": f"Task {task_id} deleted successfully"}
    except KeyError:
        raise HTTPException(status_code=404, detail="Task not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8848)
