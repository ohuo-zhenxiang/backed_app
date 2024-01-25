import redis
import json
from settings import REDIS_CONFIG


# 只能用with的方法使用， 方便上下文管理
class RedisModule:
    def __init__(self, host=REDIS_CONFIG["redis_host"], port=REDIS_CONFIG["redis_port"],
                 password=REDIS_CONFIG["redis_password"], db=REDIS_CONFIG["redis_db"]):
        self._pool = redis.ConnectionPool(host=host, port=port, password=password, decode_responses=False, db=db)
        self._query_redis = None

    def __enter__(self):
        try:
            self._query_redis = redis.Redis(connection_pool=self._pool)
            return self
        except redis.RedisError as e:
            raise f"Failed to connect to Redis: {e}"

    def __exit__(self, exc_type, exc_value, traceback):
        if self._query_redis:
            self._query_redis.close()

    def set(self, key, value):
        self._query_redis.set(key, value)

    def get(self, key):
        if not self._query_redis.get(key):
            return None
        return self._query_redis.get(key)

    def delete(self, key) -> bool:
        if not self._query_redis.get(key):
            return False
        self._query_redis.delete(key)
        return True

    def set_json(self, key, value):
        # 存储json序列化后的字符串
        self._query_redis.set(key, json.dumps(value))

    def get_json(self, key):
        # 返回数据之前先解序列化
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

    def publish(self, channel, msg):
        self._query_redis.publish(channel, msg)
        return True

    def subscribe(self, channel):
        pub = self._query_redis.pubsub()
        pub.subscribe(channel)
        pub.parse_response()
        return pub


if __name__ == "__main__":
    with RedisModule() as r:
        res = r.keys()
        print(res)
