from sqlalchemy.orm import Session
import crud, schemas
import settings
from db.base_class import Base
from db.session import engine


def init_db(db: Session) -> None:

    Base.metadata.create_all(bind=engine)

    user = crud.crud_user.get_by_phone(db, phone=settings.FIRST_SUPERUSER)
    if not user:
        user_in = schemas.UserCreate(phone=settings.FIRST_SUPERUSER, password=settings.FIRST_SUPERUSER_PASSWORD, is_superuser=True)
        user = crud.crud_user.create(db, obj_in=user_in)
