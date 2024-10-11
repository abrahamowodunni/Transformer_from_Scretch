from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4, # i could explore flexible learning rate
        "seq_len": 350,
        "d_model": 512,
        "data_path": "C:\Users\mayuo\OneDrive\Documents\Machine Learning by Abraham\Transformer_from_Scretch\Train.csv",  # Path to your local CSV file
        "lang_src": "Yoruba",                      # Source language (Yoruba)
        "lang_tgt": "English",                     # Target language (English)
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
