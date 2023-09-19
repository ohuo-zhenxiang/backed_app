from log_scheduler import RedisChannel
from loguru import logger
import pickle
from pprint import pprint
import json

while True:
    with RedisChannel() as r:
        # data1 = r.subscribe("test1").parse_response()[2]
        # data1 = json.loads(data1)
        # print(data1)
        # data2 = r.subscribe("test2").parse_response()[2]
        # data2 = json.loads(data2)
        # print(data2)
        for channel in ['test1', 'test2']:
            data = r.subscribe(channel).parse_response()
            print(data)
