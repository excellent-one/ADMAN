"""Checkpoint helpers shared by all training entry points."""
import csv
import sys
import types
from pathlib import Path

import torch

def _torch_load(path, device):
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


def checkpoint_state_dict(path, device):
    loaded = _torch_load(path, device)
    if isinstance(loaded, torch.nn.Module):
        try:
            state = loaded.state_dict()
        except AttributeError:
            # Some old DSAN pickles registered ``LMMD_loss`` (a plain helper
            # object) in ``_modules``.  It has no parameters; omit it while
            # collecting the actual network state, then restore the object.
            modules = getattr(loaded, '_modules', None)
            if not isinstance(modules, dict):
                raise
            valid = {key: value for key, value in modules.items()
                     if isinstance(value, torch.nn.Module)}
            loaded._modules = valid
            try:
                state = loaded.state_dict()
            finally:
                loaded._modules = modules
    elif isinstance(loaded, dict):
        state = loaded
        for key in ('state_dict', 'model_state_dict', 'model', 'net'):
            candidate = state.get(key)
            if isinstance(candidate, torch.nn.Module):
                state = candidate.state_dict()
                break
            if isinstance(candidate, dict):
                state = candidate
                break
    else:
        state = None
    if not isinstance(state, dict):
        raise TypeError('checkpoint file must be a state_dict, a checkpoint dict or a full model: {}'.format(path))
    return {key[7:] if key.startswith('module.') else key: value for key, value in state.items()}


def load_checkpoint(model, path, device, strict=False):
    """Load a checkpoint, including legacy teacher/student baseline models."""
    state = checkpoint_state_dict(path, device)
    model_state = model.state_dict()
    candidates = [('original', state)]
    for source_prefix in ('student_net.', 'teacher_net.'):
        head_prefix = source_prefix.replace('_net.', '_fc.')
        remapped = {}
        for key, value in state.items():
            if key.startswith(source_prefix):
                remapped['sharedNet.' + key[len(source_prefix):]] = value
            elif key.startswith(head_prefix):
                remapped['cls_fc.' + key[len(head_prefix):]] = value
        if remapped:
            candidates.append((source_prefix[:-1], remapped))

    def compatible(candidate):
        return {
            key: value for key, value in candidate.items()
            if key in model_state and getattr(value, 'shape', None) == model_state[key].shape
        }

    name, selected = max(candidates, key=lambda item: len(compatible(item[1])))
    selected = compatible(selected)
    if not selected:
        raise RuntimeError('no parameter in this checkpoint is loadable with a matching shape for the current model: {}'.format(path))
    incompatible = model.load_state_dict(selected, strict=strict)
    return incompatible, name, len(selected), len(model_state)


def pretrained_path(checkpoint):
    """Return the configured pretrained path when loading is enabled.

    The preferred form is ``{enabled: bool, model_path: path}``. A legacy
    string path is also accepted and treated as enabled for compatibility.
    """
    value = (checkpoint or {}).get("pretrained")
    if isinstance(value, str):
        return value
    if not isinstance(value, dict) or not value.get("enabled", False):
        return None
    return value.get("model_path")


def load_pretrained(model, checkpoint, device, strict=False):
    """Load a configured pretrained state dict, or return ``model`` unchanged."""
    configured = pretrained_path(checkpoint)
    if not configured:
        return model
    path = Path(configured)
    if not path.is_file():
        raise FileNotFoundError(
            "pretrained weights are enabled but the file does not exist: {}. Check checkpoint.pretrained.model_path".format(path)
        )
    load_checkpoint(model, path, device, strict=strict)
    print("Loaded pretrained weights: {}".format(path))
    return model


def save_topk_checkpoint(model, records, accuracy, step, output_dir, limit, filename):
    """Save one candidate, prune to Top-K and rewrite the ranking manifest."""
    limit = max(1, int(limit))
    if len(records) >= limit and accuracy <= records[0][0]:
        return False
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    path = output / filename
    torch.save(model.state_dict(), path)
    records.append((float(accuracy), int(step), path))
    records.sort(key=lambda item: (item[0], item[1]))
    while len(records) > limit:
        _, _, old_path = records.pop(0)
        if old_path.exists():
            old_path.unlink()
    with (output / 'manifest.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['rank', 'accuracy_percent', 'step', 'checkpoint'])
        for rank, (score, saved_step, saved_path) in enumerate(reversed(records), 1):
            writer.writerow([rank, score, saved_step, saved_path.name])
    return True
