from typing import List

from pydantic import BaseModel


class Behavior(BaseModel):
    behavior_box: List[int] = []
    behavior_type: str = ''
    behavior_score: float = 0.


class PersonBehavior(BaseModel):
    calling: List[Behavior] = []
    smoking: List[Behavior] = []


class PersonPose(BaseModel):
    pose_coords: List[List[int]] = []
    pose_scores: List[float] = []


class Person(BaseModel):
    person_id: int = 0
    person_box: List[int] = []
    person_score: float = 0.0
    person_behaviors: PersonBehavior = PersonBehavior()
    person_poses: PersonPose = PersonPose()


if __name__ == "__main__":
    from pprint import pprint
    p = Person()
    p_dict = p.model_dump()
    p_json = p.model_dump_json()
    pprint(p)
    print(type(p))