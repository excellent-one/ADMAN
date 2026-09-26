"""YAML configuration loading shared by training and evaluation commands."""
from pathlib import Path
import copy
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_config(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open('r', encoding='utf-8') as handle:
        cfg = yaml.safe_load(handle) or {}
    cfg['_config_path'] = str(path)
    return resolve_paths(cfg)

def resolve_paths(cfg):
    result = copy.deepcopy(cfg)

    def visit(values):
        for key, value in list(values.items()):
            if isinstance(value, dict):
                visit(value)
            elif isinstance(value, str) and (key.endswith('root') or key.endswith('_dir')
                                             or key.endswith('_path') or key in ('root', 'output_dir')):
                if value and not Path(value).is_absolute():
                    values[key] = str(PROJECT_ROOT / value)

    for section in ('data', 'checkpoint', 'evaluation'):
        values = result.get(section, {})
        if isinstance(values, dict):
            visit(values)
    return result

def get_device(cfg):
    import torch
    requested = cfg.get('device', 'cuda')
    return torch.device('cuda' if requested == 'cuda' and torch.cuda.is_available() else 'cpu')
