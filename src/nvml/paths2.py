import pyprojroot

BASE_DIR = pyprojroot.find_root(pyprojroot.has_dir(".git"))


class Settings:
    long_term_storage: DirectoryPath
    artifacts: DirectoryPath

    @computed_field
    @property
    def path_to_figures(self):
        return self.artifacts / "figures"

    @computed_field
    @property
    def path_to_data(self):
        return self.artifacts / "data"

    @computed_field
    @property
    def path_to_results(self):
        return self.artifacts / "data"

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
    )
