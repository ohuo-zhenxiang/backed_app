from datetime import datetime
from typing import Any, List

from sqlalchemy.orm import Session

from crud.crud_base import CRUDBase
from models import HumanTask, HumanRecord
from schemas import HumanTaskCreate, HumanTaskUpdate


class CRUDHumanTask(CRUDBase[HumanTask, HumanTaskCreate, HumanTaskUpdate]):
    def create_human_task(self, db: Session, task_name: str, interval_seconds: int, start_time: str, end_time: str,
                          capture_path: str, task_token: str, status: str, expand_tasks: List[str]) -> Any:
        db_obj = HumanTask(task_name=task_name, expand_tasks=expand_tasks,
                           interval_seconds=interval_seconds, start_time=start_time,
                           end_time=end_time, capture_path=capture_path, task_token=task_token, status=status,
                           created_time=datetime.now().replace(microsecond=0))
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def delete_human_task_by_token(self, db: Session, task_token: str):
        db.query(HumanTask).filter(HumanTask.task_token == task_token).delete()
        db.query(HumanRecord).filter(HumanRecord.task_token == task_token).delete()
        db.commit()
        return

    def get_HumanTask_by_TaskName(self, db: Session, task_name: str):
        return db.query(HumanTask).filter(HumanTask.task_name == task_name).first()


crud_human_task = CRUDHumanTask(HumanTask)
