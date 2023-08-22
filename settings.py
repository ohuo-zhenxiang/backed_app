ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8
SQLALCHEMY_DATABASE_URI: str = 'postgresql://postgres:postgres@localhost:5432/fastapi'

FIRST_SUPERUSER: str = "admin"
FIRST_SUPERUSER_PASSWORD: str = "admin"

UPLOAD_DIR = "./FaceImageData"

TASK_RECORD_DIR = "./TaskRecord"

# redis config
REDIS_CONFIG = {"redis_host": "127.0.0.1",
                "redis_port": 6379,
                "redis_password": "redis",
                "redis_db": 6}
