import json

from sqlalchemy.orm import Session

import crud
import schemas
import settings
from db.base_class import Base
from db.session import engine

init_routes = [
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

    # create super-user
    admin_user = crud.crud_user.get_by_phone(db, phone=settings.FIRST_SUPERUSER)
    if not admin_user:
        admin_user_in = schemas.UserCreate(phone=settings.FIRST_SUPERUSER,
                                           password=settings.FIRST_SUPERUSER_PASSWORD,
                                           is_superuser=True,
                                           routes=json.dumps(init_routes),
                                           role='admin',
                                           )
        admin_user = crud.crud_user.create(db, obj_in=admin_user_in)

    # create user-user
    user_user = crud.crud_user.get_by_phone(db, phone='user')
    if not user_user:
        user_user_in = schemas.UserCreate(phone='user',
                                          password='user',
                                          role='user',
                                          routes=json.dumps(init_routes),
                                          )
        user_user = crud.crud_user.create(db, obj_in=user_user_in)
