from pathlib import Path

import pyprojroot
from pydantic import DirectoryPath, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = pyprojroot.find_root(pyprojroot.has_dir(".git"))


class Settings(BaseSettings):
    long_term_storage: DirectoryPath
    artifacts: DirectoryPath
    # keep this mathcing the .env only

    @computed_field
    @property
    def path_to_figures(self) -> Path:
        return self.artifacts / "figures"

    @computed_field
    @property
    def path_to_data(self) -> Path:
        return self.artifacts / "data"

    @computed_field
    @property
    def path_to_results(self) -> Path:
        return self.artifacts / "data"

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()  # pyright: ignore[reportCallIssue]
