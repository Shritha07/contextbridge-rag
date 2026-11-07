
from dotenv import find_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    db_url: str
    cohere_key: str = ""

    model_config = SettingsConfigDict(
        env_file=find_dotenv(usecwd=True) or ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )
