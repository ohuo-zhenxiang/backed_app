import os
from db.init_db import init_db
from db.session import SessionLocal
from settings import UPLOAD_DIR, TASK_RECORD_DIR


def init() -> None:
    db = SessionLocal()
    init_db(db)


def main() -> None:
    init()


if __name__ == "__main__":
    upload_dir = UPLOAD_DIR
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    task_record_dir = TASK_RECORD_DIR
    if not os.path.exists(task_record_dir):
        os.makedirs(task_record_dir)
    # 初始化数据库
    main()
