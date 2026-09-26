"""Configuration-driven optimizer construction shared by all trainers."""
from collections.abc import Mapping

import torch


SUPPORTED_OPTIMIZERS = ('sgd', 'adam', 'adamw')


def _optimizer_settings(training):
    settings = training.get('optimizer', {})
    if settings is None:
        settings = {}
    elif isinstance(settings, str):
        settings = {'name': settings}
    elif not isinstance(settings, Mapping):
        raise TypeError('training.optimizer must be a dict or an optimizer name string')
    return settings


def optimizer_name(training):
    """Return a normalized optimizer name, defaulting to the original SGD."""
    name = str(_optimizer_settings(training).get('name', 'sgd')).lower()
    name = name.replace('-', '').replace('_', '')
    if name not in SUPPORTED_OPTIMIZERS:
        raise ValueError(
            'unsupported optimizer {!r}, valid choices are: {}'.format(
                name, ', '.join(SUPPORTED_OPTIMIZERS)))
    return name


def _betas(settings):
    betas = settings.get('betas', (0.9, 0.999))
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError('training.optimizer.betas must contain two numbers')
    return float(betas[0]), float(betas[1])


def build_optimizer(parameters, training, lr):
    """Build SGD, Adam or AdamW while retaining legacy YAML compatibility.

    ``momentum`` and ``weight_decay`` are read from ``training.optimizer``.
    Their former locations directly under ``training`` remain supported.
    """
    settings = _optimizer_settings(training)
    name = optimizer_name(training)
    weight_decay = float(settings.get(
        'weight_decay', training.get('weight_decay', 0.0)))

    if name == 'sgd':
        momentum = float(settings.get(
            'momentum', training.get('momentum', 0.9)))
        return torch.optim.SGD(
            parameters,
            lr=float(lr),
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=float(settings.get('dampening', 0.0)),
            nesterov=bool(settings.get('nesterov', False)),
        )

    common = {
        'lr': float(lr),
        'betas': _betas(settings),
        'eps': float(settings.get('eps', 1e-8)),
        'weight_decay': weight_decay,
        'amsgrad': bool(settings.get('amsgrad', False)),
    }
    if name == 'adam':
        return torch.optim.Adam(parameters, **common)
    return torch.optim.AdamW(parameters, **common)
