import os
from pathlib import Path

list_of_files = [
    "src/__init__.py",
    "src/components/__init__.py",
    "src/utils.py",
    "src/exception.py",
    "src/logger.py",
    "src/components/data_ingestion.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "src/models/__init__.py",
    "app.py",
    "templates/index.html",
    "Dockerfile",
    "notebooks/trails.ipynb",
    "setup.py",
    "requirements.txt"
]

for file_path in list_of_files:
    filepath = Path(file_path)
    file_dir, file_path = os.path.split(filepath)
    if file_dir!='':
        os.makedirs(file_dir,exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath,"w") as f:
            pass
    else:
        pass