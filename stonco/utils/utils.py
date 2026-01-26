import os, joblib, json
import torch

def save_model(model, path, filename='model.pt'):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, filename))

def load_model_state_dict(path, map_location='cpu'):
    return torch.load(os.path.join(path, 'model.pt'), map_location=map_location)

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    import json
    with open(path, 'r') as f:
        return json.load(f)
