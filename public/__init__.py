from pathlib import Path
from typing import Annotated, Literal

import pyprojroot
from pydantic import DirectoryPath, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource
from pytask import Product

TaskProduct = Annotated[Path, Product]
