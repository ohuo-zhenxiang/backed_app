from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ProcessPoolExecutor
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger
from datetime import datetime
import redis
import json
import pickle
import copy

# 自定义不同的日志文件和格式

process_executor = ProcessPoolExecutor(10)
scheduler = BackgroundScheduler(timezone='Asia/Shanghai', executors={'process': process_executor})


class RedisChannel:
    def __init__(self):
        self.connection_pool = redis.ConnectionPool(host='127.0.0.1', port=6379, password='redis', db=4)
        self.redis_conn = None

    def __enter__(self):
        self.redis_conn = redis.Redis(connection_pool=self.connection_pool)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.redis_conn:
            self.redis_conn.close()

    def publish(self, channel, msg):
        self.redis_conn.publish(channel, msg)
        return True

    def subscribe(self, channel):
        pub = self.redis_conn.pubsub()
        pub.subscribe(channel)
        pub.parse_response()
        return pub

    def get_channels(self):
        return self.redis_conn.pubsub_channels()


def task(i):
    from loguru import logger
    logger.remove()
    task_logger = logger.bind(task_name=f'task{i}')
    task_logger.add(f'task{i}.log', enqueue=True)
    task_logger.info(f'task{i}')


def task_task(task_id):
    redis_channel = RedisChannel()
    t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    msg = {"task_id": task_id, "time": t,  "status": ""}
    if task_id.endswith('1'):
        msg["status"] = "success"
        with redis_channel as r:
            r.publish('test1', json.dumps(msg))
        logger.success(f'{task_id}')
    else:
        msg["status"] = "error"
        with redis_channel as r:
            r.publish('test2', json.dumps(msg))
        logger.error(f"{task_id}")


if __name__ == '__main__':
    scheduler.add_job(task_task, args=['pubsub-1'], trigger=IntervalTrigger(seconds=2), id='task1',
                      replace_existing=True, executor="process")
    scheduler.add_job(task_task, args=['pubsub-2'], trigger=IntervalTrigger(seconds=5), id='task2',
                      replace_existing=True, executor="process")

    scheduler.start()
    while True:
        pass
