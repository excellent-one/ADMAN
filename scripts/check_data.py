"""Validate that every data root/domain declared by a YAML file exists."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config import load_config

def check(config_path):
    cfg = load_config(config_path); missing = []
    for section in ('data', 'evaluation'):
        values = cfg.get(section, {})
        for key, value in values.items():
            if ('root' in key or key in ('domain', 'test_domain', 'target_domain', 'source_domain_high', 'source_domain_low', 'pet_domain_high', 'pet_domain_low', 'hr_domain_high', 'hr_domain_low')) and isinstance(value, str):
                path = Path(value)
                if key.endswith('domain') or key.startswith('source_domain') or key.startswith('pet_domain') or key.startswith('hr_domain'):
                    roots = [v for k, v in values.items() if k.endswith('root')]
                    if not any((Path(root) / value).exists() for root in roots): missing.append((key, value))
                elif not path.exists(): missing.append((key, value))
    if missing:
        for key, value in missing: print('MISSING', key, value)
        return 1
    print('OK:', config_path); return 0

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('configs', nargs='+')
    raise SystemExit(max(check(c) for c in p.parse_args().configs))
