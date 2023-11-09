import os
from glob import glob

db_paths = glob("./**.db")
for path in db_paths:
    os.remove(path=path)

print("[INFO] All `db` files removed")
