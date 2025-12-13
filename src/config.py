from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    bot_token: str
    api_base_url: str = "http://localhost:8000"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
