from typing import Any

from sqlalchemy.ext.declarative import as_declarative, declared_attr


# as_declarative装饰器会将Base类转换为SQLALchemy的declarative base类，简洁orm映射类
@as_declarative()
class Base:
    id: Any
    __name__: str

    # Generate __tablename__ automatically, declared_attr装饰器动态创建一个__tablename__属性，值是类名的小写形式
    @declared_attr
    def __tablename__(cls) -> str:
        return cls.__name__.lower()



