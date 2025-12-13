from pydantic_settings import BaseSettings, SettingsConfigDict


def _build_db_url(
    user: str,
    password: str,
    host: str,
    port: int,
    db: str,
):
    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"


class Settings(BaseSettings):
    bot_token: str
    api_base_url: str = "http://localhost:8000"
    admin_token: str
    user_token: str

    db_user: str = "postgres"
    db_password: str = "postgres"
    db_host: str = "127.0.0.1"
    db_port: int = 5432
    db_name: str = "legal_rag"
    database_url: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    def __init__(self, **data):
        super().__init__(**data)
        if not self.database_url:
            self.database_url = _build_db_url(
                self.db_user,
                self.db_password,
                self.db_host,
                self.db_port,
                self.db_name,
            )


settings = Settings()
