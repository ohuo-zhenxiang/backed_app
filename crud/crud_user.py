from typing import Any, Dict, Optional, Union

from sqlalchemy.orm import Session

from crud.crud_base import CRUDBase
from models.user import User
from schemas.user import UserCreate, UserUpdate
from core.security import get_password_hash, verify_password


class CRUDUser(CRUDBase[User, UserCreate, UserUpdate]):
    def get_by_phone(self, db: Session, *, phone: str) -> Optional[User]:
        return db.query(User).filter(User.phone == phone).first()

    def get_by_id(self, db: Session, *, user_id: int) -> Optional[User]:
        return db.query(User).filter(User.id == user_id).first()

    def create(self, db: Session, *, obj_in: UserCreate) -> User:
        db_obj = User(phone=obj_in.phone, hashed_password=get_password_hash(obj_in.password),
                      full_name=obj_in.full_name, is_superuser=obj_in.is_superuser,
                      routes=obj_in.routes, role=obj_in.role)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(self, db: Session, *, db_obj: User, obj_in: Union[UserUpdate, Dict[str, Any]]) -> User:
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)
        if update_data["password"]:
            hashed_password = get_password_hash(update_data["password"])
            del update_data["password"]
            update_data["hashed_password"] = hashed_password
        return super().update(db, db_obj=db_obj, obj_in=update_data)

    def authenticate(self, db: Session, *, phone: str, password: str) -> Optional[User]:
        user = self.get_by_phone(db, phone=phone)
        if not user:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user

    def is_activate(self, user: User) -> bool:
        return user.is_activate

    def is_superuser(self, user: User) -> bool:
        return user.is_superuser


crud_user = CRUDUser(User)
