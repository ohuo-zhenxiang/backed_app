from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ProcessPoolExecutor
from apscheduler.triggers.interval import IntervalTrigger

import copy

# from loguru import logger
#
# # 创建两个不同的 Loguru logger 实例
# a_logger = copy.deepcopy(logger)
# a_logger.add('alog.log', enqueue=True)
# a_logger.bind(name='a')
#
# b_logger = copy.deepcopy(logger)
# b_logger.add('blog.log', enqueue=True)
# b_logger.bind(name='b')

# 自定义不同的日志文件和格式

process_executor = ProcessPoolExecutor(10)
scheduler = BackgroundScheduler(timezone='Asia/Shanghai', executors={'process': process_executor})


def task(i):
    from loguru import logger
    logger.remove()
    task_logger = logger.bind(task_name=f'task{i}')
    task_logger.add(f'task{i}.log', enqueue=True)
    task_logger.info(f'task{i}')


if __name__ == '__main__':
    scheduler.add_job(task, args=[1], trigger=IntervalTrigger(seconds=5), id='task1',
                      replace_existing=True, executor="process")
    scheduler.add_job(task, args=[2], trigger=IntervalTrigger(seconds=5), id='task2',
                      replace_existing=True, executor="process")
    scheduler.add_job(task, args=[3], trigger=IntervalTrigger(seconds=5), id='task3',
                      replace_existing=True, executor="process")

    scheduler.start()
    while True:
        pass
