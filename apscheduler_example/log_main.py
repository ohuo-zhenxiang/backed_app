from loguru import logger

# 创建不同的日志记录器实例并设置标识
logger.add("my_log.log", format="{time:YY-MM-DD HH:mm:ss} | {level} | {extra[name]} | {message}", enqueue=True)

logger1 = logger.bind(name="Logger 1")
logger2 = logger.bind(name="Logger 2")

# 使用不同的日志记录器实例记录日志
logger1.error("This is from Logger 1")
logger2.error("This is from Logger 2")

# 输出：
# [INFO] [Logger 1] This is from Logger 1
# [INFO] [Logger 2] This is from Logger 2
