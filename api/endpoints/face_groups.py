from typing import Any, List

from fastapi import APIRouter, Depends, status, Form
from fastapi.responses import JSONResponse
from fastapi_pagination.ext.sqlalchemy import select
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

import crud
import models
import schemas
from api import deps

groups_logger = logger.bind(name="groups")
router = APIRouter()

# 这里到底要不要用分页功能捏？---->暂时就先不用了
'''@router.get("/get_groups", response_model=Page[schemas.GroupSelect])
async def get_groups(db: Session = Depends(deps.get_db)) -> Any:
    query = db.query(models.Group).order_by(desc(models.Group.id))
    return paginate(query)'''


@router.get("/get_groups", response_model=List[schemas.GroupSelect])
async def get_groups(db: Session = Depends(deps.get_db)) -> Any:
    # 创建子查询来统计每个分组的成员人数
    subquery = (
        select(
            models.Group.id,
            func.count(models.Face.id).label("member_count")
        )
        .join(models.Group.members)
        .group_by(models.Group.id)
        .subquery()
    )
    # 主查询连接 Group 表和子查询，并选择需要的列
    result = (
        db.query(models.Group.id, models.Group.name, models.Group.description,
                 func.coalesce(subquery.c.member_count, 0).label("member_count"))
        .outerjoin(subquery, models.Group.id == subquery.c.id)
        .order_by(desc(models.Group.id))
        .all()
    )
    return result


@router.get("/get_group_ids")
async def get_group_ids(db: Session = Depends(deps.get_db)):
    # 创建子查询来统计每个分组的成员人数
    subquery = (
        select(
            models.Group.id,
            func.count(models.Face.id).label("member_count")
        )
        .join(models.Group.members)
        .group_by(models.Group.id)
        .subquery()
    )
    # 主查询连接 Group 表和子查询，并选择需要的列
    result = (
        db.query(models.Group.id, models.Group.name,
                 func.coalesce(subquery.c.member_count, 0).label("member_count"))
        .outerjoin(subquery, models.Group.id == subquery.c.id)
        .order_by(desc(models.Group.id))
        .all()
    )

    result = [{"value": i[0], "label": i[1], "disabled": False if i[2] > 0 else True} for i in result]
    return result


@router.post("/create_group")
async def create_group(post_group: schemas.GroupCreate, db: Session = Depends(deps.get_db)):
    """
    Create a new group
    """
    post_group = post_group.dict()
    name = post_group["name"]
    desc = post_group.get("description", "")
    a = crud.crud_group.get_by_name(db, name)
    if a:
        groups_logger.warning(f"AddGroup {name} already exists")
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,
                            content={"message": f"Group {name}  already exists"})
    else:
        b = crud.crud_group.create_group(db, name, desc)
        if b:
            groups_logger.success(f"AddGroup {name}")
            return JSONResponse(status_code=status.HTTP_200_OK, content={"message": f"Group {name} created"})


@router.delete("/delete_group/{group_id}")
async def delete_group(group_id: int, db: Session = Depends(deps.get_db)):
    """
    Delete a group
    """
    group = crud.crud_group.get_group_by_id(db, group_id)
    if group:
        crud.crud_group.remove_group(db, group)
        groups_logger.success(f"DeleteGroup {group_id}")
        return JSONResponse(status_code=status.HTTP_200_OK, content={"message": f"Group {group_id} deleted"})
    else:
        groups_logger.warning(f"DeleteGroup {group_id} does not exist")
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,
                            content={"message": f"Group {group_id} does not exist"})


@router.put("/update_group/{group_id}")
async def update_group(group_id: int, db: Session = Depends(deps.get_db), group_name: str = Form(),
                       group_description: str = Form()):
    """
    Update a group
    """
    group = crud.crud_group.get_group_by_id(db, group_id)
    if group:
        crud.crud_group.update_group_by_id(db, group_id=group_id, name=group_name, desc=group_description)
        groups_logger.success(f"UpdateGroup {group_id}")
        return JSONResponse(status_code=status.HTTP_200_OK, content={"message": f"Group {group_id} updated"})
    else:
        groups_logger.warning(f"UpdateGroup {group_id} does not exist")
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,
                            content={"message": f"Group {group_id} does not exist"})


@router.get("/get_group_members/{group_id}", response_model=List[int])
async def get_group_members(group_id: int, db: Session = Depends(deps.get_db)) -> Any:
    """
    Get all group members
    """
    group = db.query(models.Group).filter(models.Group.id == group_id).first()
    if group:
        result = db.query(models.Face.id.label('value')).join(models.Group.members).filter(
            models.Group.id == group_id).all()
        result = [int(i[0]) for i in result]
        return result


@router.get("/get_all_members", response_model=List[schemas.MembersInGroup])
async def get_all_members(db: Session = Depends(deps.get_db)) -> Any:
    """
    Get all group members
    """
    result = db.query(models.Face.id.label('value'), func.concat(models.Face.name, '-', models.Face.phone)
                      .label('label')).order_by(models.Face.id).all()
    return result


class UpdateGroup(BaseModel):
    member_ids: List[int]
    group_name: str
    group_description: str


@router.put("/update_group_members/{group_id}")
async def update_group_members(group_id: int, update_data: UpdateGroup,
                               db: Session = Depends(deps.get_db)):
    """
    Update group members
    """
    print(update_data)
    a = db.query(models.Group).filter(models.Group.name == update_data.group_name, models.Group.id != group_id).first()
    if a:
        return JSONResponse(status_code=status.HTTP_409_CONFLICT, content={"error": "Group name already exists"})

    group = db.query(models.Group).filter(models.Group.id == group_id).first()
    if group:
        group.members.clear()
        new_members = db.query(models.Face).filter(models.Face.id.in_(update_data.member_ids)).all()
        group.members.extend(new_members)
        group.name = update_data.group_name
        group.description = update_data.group_description
        db.commit()
        groups_logger.success(f"UpdatedGroupMembers {group_id}")
        return JSONResponse(status_code=status.HTTP_200_OK,
                            content={"message": f"Group {group_id}"})

# @router.post("/add_group_members/{group_id}")
# async def add_group_members(group_id: int, member_ids: List[int], db: Session = Depends(deps.get_db)):
#     """
#     Add members to group
#     """
#     print(member_ids)
#     print(group_id)
#     group = db.query(models.Group).filter(models.Group.id == group_id).first()
#     if group:
#         faces = db.query(models.Face).filter(models.Face.id.in_(member_ids)).all()
#         group.members.extend(faces)
#         db.commit()
#         return JSONResponse(status_code=status.HTTP_200_OK,
#                             content={"message": f"Members added to group {group.name} successfully"})
#     else:
#         return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,
#                             content={"message": f"Group {group_id} does not exist"})
