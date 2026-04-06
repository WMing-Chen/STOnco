import json
import numbers
import os
import joblib

DEFAULT_GNN_HIDDEN = (256, 128, 64)


def _parse_positive_int(value, field_name):
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer, got bool")
    if isinstance(value, str):
        value = value.strip()
        if value == '':
            raise ValueError(f"{field_name} must not be empty")
        try:
            value = int(value)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a positive integer, got: {value}") from exc
    elif isinstance(value, numbers.Integral):
        value = int(value)
    else:
        raise ValueError(f"{field_name} must be a positive integer, got: {value}")
    if int(value) <= 0:
        raise ValueError(f"{field_name} must be > 0, got: {value}")
    return int(value)


def _parse_gnn_hidden_value(value, field_name):
    if isinstance(value, str):
        s = value.strip()
        if s == '':
            raise ValueError(f"{field_name} must not be empty")
        if ',' in s:
            parts = [p.strip() for p in s.split(',') if p.strip() != '']
            if not parts:
                raise ValueError(f"{field_name} must not be empty")
            return [_parse_positive_int(part, f'{field_name}[{idx}]') for idx, part in enumerate(parts)]
        return _parse_positive_int(s, field_name)
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError(f"{field_name} must not be an empty list")
        return [_parse_positive_int(part, f'{field_name}[{idx}]') for idx, part in enumerate(value)]
    return _parse_positive_int(value, field_name)


def normalize_gnn_hidden(hidden=None, gnn_hidden=None, num_layers=None, default_gnn_hidden=DEFAULT_GNN_HIDDEN):
    has_hidden = hidden is not None
    has_gnn_hidden = gnn_hidden is not None
    if has_hidden and has_gnn_hidden:
        raise ValueError("Do not pass both 'hidden' and 'GNN_hidden'; use only 'GNN_hidden'.")

    if has_gnn_hidden:
        raw_hidden = gnn_hidden
        field_name = 'GNN_hidden'
    elif has_hidden:
        raw_hidden = hidden
        field_name = 'hidden'
    else:
        raw_hidden = list(default_gnn_hidden)
        field_name = 'GNN_hidden'

    explicit_num_layers = num_layers is not None
    parsed_num_layers = _parse_positive_int(num_layers, 'num_layers') if explicit_num_layers else None
    parsed_hidden = _parse_gnn_hidden_value(raw_hidden, field_name)

    if isinstance(parsed_hidden, list):
        if explicit_num_layers and parsed_num_layers != len(parsed_hidden):
            raise ValueError(
                f"num_layers ({parsed_num_layers}) does not match len(GNN_hidden) ({len(parsed_hidden)})."
            )
        return parsed_hidden, len(parsed_hidden)

    if not explicit_num_layers:
        raise ValueError("num_layers must be provided when GNN_hidden/hidden is a scalar.")

    return [parsed_hidden] * parsed_num_layers, parsed_num_layers


def normalize_gnn_config(cfg, default_gnn_hidden=DEFAULT_GNN_HIDDEN):
    cfg = dict(cfg)
    hidden_list, effective_num_layers = normalize_gnn_hidden(
        hidden=cfg.get('hidden'),
        gnn_hidden=cfg.get('GNN_hidden'),
        num_layers=cfg.get('num_layers'),
        default_gnn_hidden=default_gnn_hidden,
    )
    cfg['GNN_hidden'] = [int(v) for v in hidden_list]
    cfg['num_layers'] = int(effective_num_layers)
    cfg.pop('hidden', None)
    return cfg

def save_model(model, path, filename='model.pt'):
    import torch

    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, filename))

def load_model_state_dict(path, map_location='cpu'):
    import torch

    return torch.load(os.path.join(path, 'model.pt'), map_location=map_location)

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    import json
    with open(path, 'r') as f:
        return json.load(f)
