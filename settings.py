import os

# PATH
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "FaceImageData")
TASK_RECORD_DIR = os.path.join(BASE_DIR, "TaskRecord")
LOGGING_DIR = os.path.join(BASE_DIR, "Log")
MODEL_DIR = os.path.join(BASE_DIR, 'Engines')
FEATURE_LIB = os.path.join(BASE_DIR, 'FeatureLib')

ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8
SQLALCHEMY_DATABASE_URI: str = 'postgresql://postgres:postgres@localhost:5432/fastapi'

FIRST_SUPERUSER: str = "admin"
FIRST_SUPERUSER_PASSWORD: str = "admin"

# redis config
REDIS_CONFIG = {"redis_host": "127.0.0.1",
                "redis_port": 6379,
                "redis_password": "redis",
                "redis_db": 4}
REDIS_URL = "redis://:redis@127.0.0.1:6379/4"
