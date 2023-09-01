import redis

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor
from apscheduler.jobstores.redis import RedisJobStore
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.jobstores.memory import MemoryJobStore
from pytz import timezone
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED

process_executor = ProcessPoolExecutor(10)
redis_connection = redis.ConnectionPool(host='127.0.0.1', port=6379, db=7, password='redis')
redis_jobstore = RedisJobStore(jobs_key='scheduler:jobs', run_times_key='scheduler:run_times',
                               connection_pool=redis_connection)

job_defaults = {'coalesce': True, 'max_instances': 8}
# setting scheduler
Scheduler = BackgroundScheduler(timezone='Asia/Shanghai',
                                executors={'process': process_executor},
                                # job_defaults=job_defaults
                                )
Scheduler.add_jobstore(redis_jobstore, 'redis')
