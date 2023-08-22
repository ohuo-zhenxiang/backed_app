import redis
import json
from settings import REDIS_CONFIG


class RedisModule:
    def __init__(self, host, port, password, db=REDIS_CONFIG["redis_db"]):
        self._pool = redis.ConnectionPool(host=host, port=port, password=password, decode_responses=True, db=db)
        self._query_redis = redis.Redis(connection_pool=self._pool)

    def set(self, key, value):
        self._query_redis.set(key, json.dumps(value))

    def get(self, key):
        if not self._query_redis.get(key):
            return dict()

        res = json.loads(self._query_redis.get(key))
        return res

    def keys(self):
        res = self._query_redis.keys()
        return res

    def is_connected(self):
        return self._query_redis.ping()

    def disconnect(self):
        self._query_redis.connection_pool.disconnect()


if __name__ == "__main__":
    redis = RedisModule(host="127.0.0.1", port=6379, password='redis')
    res = redis.keys()
    print(res)
