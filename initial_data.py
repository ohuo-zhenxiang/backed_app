import os
from typing import List

from db.init_db import init_db
from db.session import SessionLocal
from settings import UPLOAD_DIR, TASK_RECORD_DIR, LOGGING_DIR


def init() -> None:
    db = SessionLocal()
    init_db(db)


def main() -> None:
    init()


def add_dir(dir_name: List[str]) -> None:
    for dir in dir_name:
        if not os.path.exists(dir):
            os.makedirs(dir)


if __name__ == "__main__":
    upload_dir = UPLOAD_DIR
    dir_list = [UPLOAD_DIR, TASK_RECORD_DIR, LOGGING_DIR]
    add_dir(dir_list)
    # 初始化数据库
    main()
