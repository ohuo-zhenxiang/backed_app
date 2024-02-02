from sqlalchemy.orm import Session
import crud, schemas
import settings
from db.base_class import Base
from db.session import engine
import json

init_permissions = [
    {
        'label': '系统总览',
        'value': 'dashboard_overview'
    },
    {
        'label': '人脸管理',
        'value': 'faceDatabase_management',
    },
    {
        'label': '抓拍导入',
        'value': 'faceDatabase_snapShots',
    },
    {
        'label': '人脸分组',
        'value': 'personGroup_basic-group',
    },
    {
        'label': '任务管理',
        'value': 'TaskManage',
    },
    {
        'label': '人脸识别任务',
        'value': 'TaskManage_face-task',
    },
    {
        'label': '人体检测任务',
        'value': 'TaskManage_human-task',
    }
]


def init_db(db: Session) -> None:
    Base.metadata.create_all(bind=engine)

    user = crud.crud_user.get_by_phone(db, phone=settings.FIRST_SUPERUSER)
    if not user:
        user_in = schemas.UserCreate(phone=settings.FIRST_SUPERUSER,
                                     password=settings.FIRST_SUPERUSER_PASSWORD,
                                     is_superuser=True,
                                     permissions=json.dumps(init_permissions),
                                     )
        user = crud.crud_user.create(db, obj_in=user_in)
