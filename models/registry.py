"""Central model registry used by config-driven experiments."""
from .adman import ADMAN


SUPERVISED_MODELS = {}
SINGLE_SOURCE_MODELS = {}
MULTI_SOURCE_MODELS = {
    'adman': ADMAN,
}


def build_model(name, num_classes, source_count):
    """Construct a model and reject using a model with the wrong experiment type."""
    name = str(name).lower()
    
    if source_count == 0:
        registry = SUPERVISED_MODELS
    else:
        registry = SINGLE_SOURCE_MODELS if source_count == 1 else MULTI_SOURCE_MODELS
        
    if name not in registry:
        valid = ', '.join(sorted(registry)) if registry else 'None'
        kind = {0: 'supervised', 1: 'single-source'}.get(source_count, 'multi-source')
        raise ValueError("model={!r} is not a {} model; choose one of: {}".format(name, kind, valid))
        
    return registry[name](num_classes=int(num_classes))
