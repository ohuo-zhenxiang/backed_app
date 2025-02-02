from datetime import datetime
from typing import Any, List

from sqlalchemy.orm import Session

from crud.crud_base import CRUDBase
from models import Task, Record
from schemas import TaskCreate, TaskUpdate


class CRUDTask(CRUDBase[Task, TaskCreate, TaskUpdate]):
    def create_task(self, db: Session, task_name: str, interval_seconds: int, start_time: str, end_time: str,
                    capture_path: str, task_token: str, status: str, associated_group_id: int, ex_detect: List[str]
                    ) -> Any:
        db_obj = Task(name=task_name, ex_detect=ex_detect,
                      interval_seconds=interval_seconds, start_time=start_time, end_time=end_time,
                      capture_path=capture_path, task_token=task_token, status=status,
                      associated_group_id=associated_group_id, created_time=datetime.now().replace(microsecond=0))
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def delete_task(self, db: Session, task_token: str):
        db.query(Task).filter(Task.task_token == task_token).delete()
        db.query(Record).filter(Record.task_token == task_token).delete()
        db.commit()
        return

    def get_by_task_name(self, db: Session, task_name: str):
        return db.query(Task).filter(Task.name == task_name).first()


crud_task = CRUDTask(Task)
