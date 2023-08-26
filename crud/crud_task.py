from typing import Optional, List, Any
from sqlalchemy.orm import Session
from crud.crud_base import CRUDBase
from models import Task
from schemas import TaskCreate, TaskUpdate


class CRUDTask(CRUDBase[Task, TaskCreate, TaskUpdate]):
    def create_task(self, db: Session, task_name: str, interval_seconds: int, start_time: str, end_time: str,
                    capture_path: str, task_token: str) -> Any:
        db_obj = Task(name=task_name, interval_seconds=interval_seconds, start_time=start_time, end_time=end_time,
                      capture_path=capture_path, task_token=task_token)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj




crud_task = CRUDTask(Task)
