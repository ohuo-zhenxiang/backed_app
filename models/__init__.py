# 每次在models下定义一个数据模型class之后，直接init创表会报错，需要在这里先import引用一下才会创表

from .user import User
from .face import Face, face_group_association
from .group import Group
from .face_task_record import Task, Record
from .camera import Camera
from .human_task_record import HumanTask, HumanRecord

