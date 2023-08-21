from celery import Celery
import time

app = Celery('celery_test_tasks', broker='redis://:redis@localhost:6379/4')


@app.task
def process_task(task_id):
    time.sleep(2)
    return {"message": f"Task {task_id} processed"}