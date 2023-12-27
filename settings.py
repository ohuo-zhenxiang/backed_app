import os

import yaml

# PATH
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "FaceImageData")
TASK_RECORD_DIR = os.path.join(BASE_DIR, "TaskRecord")
LOGGING_DIR = os.path.join(BASE_DIR, "Log")
MODEL_DIR = os.path.join(BASE_DIR, 'Engines')
FEATURE_LIB = os.path.join(BASE_DIR, 'FeatureLib')

ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8

FIRST_SUPERUSER: str = "admin"
FIRST_SUPERUSER_PASSWORD: str = "admin"

with open(os.path.join(BASE_DIR, 'config.yml'), 'r') as f:
    config = yaml.safe_load(f)

redis_config = config['redis']
postgresql_config = config['postgresql']
app_config = config['app']
REC_THRESHOLD = float(config['reg_threshold'])

SQLALCHEMY_DATABASE_URI: str = f"postgresql://{postgresql_config['username']}:{postgresql_config['password']}@{postgresql_config['host']}:{postgresql_config['port']}/{postgresql_config['database']}"
# redis config
REDIS_CONFIG = {"redis_host": redis_config['host'],
                "redis_port": redis_config['port'],
                "redis_password": redis_config['password'],
                "redis_db": redis_config['db']}
REDIS_URL: str = f"redis://:{redis_config['password']}@{redis_config['host']}:{redis_config['port']}/{redis_config['db']}"

APP_PORT: int = app_config['port']
APP_HOST: str = app_config['host']
