import redis
from apscheduler.executors.pool import ProcessPoolExecutor
from apscheduler.jobstores.redis import RedisJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from pytz import timezone

from settings import REDIS_CONFIG

process_executor = ProcessPoolExecutor(10)
redis_connection = redis.ConnectionPool(host=REDIS_CONFIG["redis_host"], port=REDIS_CONFIG["redis_port"], db=7,
                                        password=REDIS_CONFIG["redis_password"])
redis_jobstore = RedisJobStore(jobs_key='scheduler:jobs', run_times_key='scheduler:run_times',
                               connection_pool=redis_connection)

job_defaults = {'coalesce': True, 'max_instances': 12, 'misfire_grace_time': 1000}
# setting scheduler
Scheduler = BackgroundScheduler(timezone=timezone('Asia/Shanghai'),
                                executors={'process': process_executor},
                                job_defaults=job_defaults
                                )

Scheduler.add_jobstore(redis_jobstore, 'redis')
