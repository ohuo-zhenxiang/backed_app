import os
from typing import List

from db.init_db import init_db
from db.session import SessionLocal
from settings import UPLOAD_DIR, TASK_RECORD_DIR, LOGGING_DIR, MODEL_DIR, FEATURE_LIB


def init() -> None:
    db = SessionLocal()
    init_db(db)


def run_init() -> None:
    dir_list = [UPLOAD_DIR, TASK_RECORD_DIR, LOGGING_DIR, MODEL_DIR, FEATURE_LIB]
    add_dir(dir_list)
    init()


def add_dir(dir_name: List[str]) -> None:
    for dir in dir_name:
        if not os.path.exists(dir):
            os.makedirs(dir)


if __name__ == "__main__":
    # 初始化数据库
    run_init()
    print('----初始化数据库完成----')
