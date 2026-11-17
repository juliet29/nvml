from pathlib import Path
from typing import Annotated, Literal

import pyprojroot
from pydantic import DirectoryPath, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource
from pytask import Product

BASE_DIR = pyprojroot.find_root(pyprojroot.has_dir(".git"))
TaskProduct = Annotated[Path, Product]


class Settings(BaseSettings):
    # machine-specific paths
    artifacts: DirectoryPath
    msd_eplus_v1: DirectoryPath
    msd_eplus_v2: DirectoryPath
    case_names_file: str

    # -- experimental config ---
    random_state: int

    # settings config -----
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        yaml_file=BASE_DIR / "config.yaml",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
        )

    # ------ figures ------
    figure_profile: Literal["journal", "slides"] = "slides"

    @computed_field
    @property
    def path_to_figures(self) -> Path:
        return self.artifacts / "figures" / self.figure_profile

    @computed_field
    @property
    def figure_ext(self) -> str:
        return "pdf" if self.figure_profile == "journal" else "svg"

    def fig_path(self, *parts: str | Path) -> Path:
        return self.path_to_figures.joinpath(*parts).with_suffix(f".{self.figure_ext}")

    # ------ other artifacts ----------
    @computed_field
    @property
    def path_to_data(self) -> Path:
        return self.artifacts / "data"

    @computed_field
    @property
    def path_to_results(self) -> Path:
        return self.artifacts / "data"


settings = Settings()  # pyright: ignore[reportCallIssue]
