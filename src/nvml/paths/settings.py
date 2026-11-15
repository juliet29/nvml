from pathlib import Path
from typing import Annotated, Literal

import pyprojroot
from pydantic import DirectoryPath, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pytask import Product

BASE_DIR = pyprojroot.find_root(pyprojroot.has_dir(".git"))
FigProduct = Annotated[Path, Product]


class Settings(BaseSettings):
    # TODO: move out of paths
    long_term_storage: DirectoryPath
    artifacts: DirectoryPath

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

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()  # pyright: ignore[reportCallIssue]
