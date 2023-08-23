from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor
from apscheduler.jobstores.redis import RedisJobStore
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from pytz import timezone
import atexit


process_executor = ProcessPoolExecutor(10)

# init scheduler
Scheduler = BackgroundScheduler(timezone=timezone('Asia/Shanghai'), executors={'process': process_executor})



