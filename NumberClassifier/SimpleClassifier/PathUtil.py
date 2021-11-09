from pathlib import Path

save_dir =  Path( './training_models')

def set_dir_path(path: str):
    save_dir = Path(path)

def get_dir_path():
    return save_dir

def create_file_path(file_name = "my_model"):
    save_path =  save_dir/ "/" / file_name / '.h5'
              
def make_dir(path: Path):
    if(path.exists()):
        return path
    else:
        path.mkdir(parents=True, exist_ok=True)

