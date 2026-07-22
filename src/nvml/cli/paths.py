import os
from pathlib import Path

import pyprojroot
from dotenv import load_dotenv

BASE_PATH = pyprojroot.find_root(pyprojroot.has_dir(".git"))

load_dotenv(Path(BASE_PATH) / ".env")

storage_path = os.getenv("PATH_TO_STORAGE")
assert storage_path
PATH_TO_STORAGE = Path(storage_path)

temp_path = os.getenv("PATH_TO_TEMP")
assert temp_path
PATH_TO_TEMP = Path(temp_path)
