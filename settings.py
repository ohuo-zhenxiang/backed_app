ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8
SQLALCHEMY_DATABASE_URI: str = 'postgresql://postgres:postgres@localhost:5432/fastapi'

FIRST_SUPERUSER: str = "admin"
FIRST_SUPERUSER_PASSWORD: str = "admin"

UPLOAD_DIR = "./data"
