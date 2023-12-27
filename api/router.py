from fastapi import APIRouter

from api.endpoints import (items, login, users, utils, faces, face_groups, face_tasks, face_records, wbs, overview, \
                           cameras, human_tasks, human_records)

api_router = APIRouter()
api_router.include_router(login.router, prefix='/login', tags=["login"])
api_router.include_router(users.router, prefix='/users', tags=['users'])
api_router.include_router(utils.router, prefix="/utils", tags=['utils'])
api_router.include_router(items.router, prefix="/items", tags=["items"])
api_router.include_router(faces.router, prefix="/faces", tags=["faces"])
api_router.include_router(face_groups.router, prefix="/groups", tags=["groups"])
api_router.include_router(face_tasks.router, prefix="/face_tasks", tags=["face_tasks"])
api_router.include_router(face_records.router, prefix="/face_records", tags=["face_records"])
api_router.include_router(wbs.router, prefix="/wbs", tags=["wbs"])
api_router.include_router(overview.router, prefix="/overview", tags=["overview"])
api_router.include_router(cameras.router, prefix="/cameras", tags=["cameras"])

api_router.include_router(human_tasks.router, prefix="/human_tasks", tags=["human_tasks"])
api_router.include_router(human_records.router, prefix="/human_records", tags=["human_records"])
