from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from redis import Redis
import atexit
import datetime
import time
import uuid
import json
from pytz import timezone


app = FastAPI()

# 初始化调度器和Redis
scheduler = BackgroundScheduler(timezone=timezone('Asia/Shanghai'))
redis = Redis(host='localhost', port=6379, db=6, password='redis')

# 关闭应用时关闭调度器
atexit.register(lambda: scheduler.shutdown())


def my_job(task_id):
    redis.hset(task_id, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "overed")
    time.sleep(2.5)


@app.post("/add_task")
async def add_task(task_name: str, interval_seconds: int, start_time: str, end_time: str):
    task_id = str(uuid.uuid4())
    trigger = IntervalTrigger(seconds=interval_seconds, start_date=start_time, end_date=end_time)
    job = scheduler.add_job(my_job, trigger, args=[task_id], id=task_id, name=task_name)
    tasks_info = {"task_id": task_id, "task_name": task_name, "interval_seconds": interval_seconds, "start_time": start_time}
    redis.hset("tasks", f"{task_id}", json.dumps(tasks_info))

    return {"message": f"Task-{task_id} added successfully"}


@app.delete("/delete_task/{task_id}")
async def delete_task(task_id: str):
    if redis.hget(task_id, "interval_seconds"):
        scheduler.remove_job(task_id)
        redis.delete(task_id)
        return {"message": "Task deleted successfully"}
    else:
        return {"message": "Task not found"}


@app.get("/list_tasks")
async def list_tasks():
    task_ids = redis.keys()
    tasks = []
    for task_id in task_ids:
        interval_seconds = int(redis.hget(task_id, "interval_seconds"))
        start_time = int(redis.hget(task_id, "start_time"))
        end_time = int(redis.hget(task_id, "end_time"))
        tasks.append({
            "task_id": task_id.decode(),
            "interval_seconds": interval_seconds,
            "start_time": start_time,
            "end_time": end_time
        })
    return {"tasks": tasks}


if __name__ == "__main__":
    import uvicorn

    # 启动调度器
    scheduler.start()
    uvicorn.run(app, host="0.0.0.0", port=8899)

