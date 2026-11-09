import os
from pathlib import Path

import pyprojroot
from dotenv import load_dotenv

BASE_PATH = pyprojroot.find_root(pyprojroot.has_dir(".git"))

load_dotenv(Path(BASE_PATH) / ".env")

storage_path = os.getenv("PATH_TO_TEMP")
assert storage_path
PATH_TO_TEMP = Path(storage_path)

VIZ_FOLDER = "viz"

PATH_TO_VIZ = PATH_TO_TEMP / VIZ_FOLDER
