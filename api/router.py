from fastapi import APIRouter
from api.endpoints import items, login, users, utils, faces, groups, tasks, records, wbs, overview

api_router = APIRouter()
api_router.include_router(login.router, prefix='/login', tags=["login"])
api_router.include_router(users.router, prefix='/users', tags=['users'])
api_router.include_router(utils.router, prefix="/utils", tags=['utils'])
api_router.include_router(items.router, prefix="/items", tags=["items"])
api_router.include_router(faces.router, prefix="/faces", tags=["faces"])
api_router.include_router(groups.router, prefix="/groups", tags=["groups"])
api_router.include_router(tasks.router, prefix="/tasks", tags=["tasks"])
api_router.include_router(records.router, prefix="/records", tags=["records"])
api_router.include_router(wbs.router, prefix="/wbs", tags=["wbs"])
api_router.include_router(overview.router, prefix="/overview", tags=["overview"])
