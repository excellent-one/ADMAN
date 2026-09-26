"""Independent LR evaluation for ADMAN."""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import build_model
from tools.data_loader import NiiDataset
from utils.checkpoint import checkpoint_state_dict
from utils.config import get_device, load_config
from utils.reporting import save_classification_artifacts


def evaluate(cfg, model_path=None, tag=None, out_dir=None, batch_size=None):
    model_path = model_path or cfg.get('checkpoint', {}).get('model_path')
    if not model_path:
        raise ValueError('provide the checkpoint path in --model-path or YAML checkpoint.model_path')
    device = get_device(cfg); d = cfg['evaluation']; batch_size = batch_size or cfg['training']['batch_size']
    dataset = NiiDataset(d['test_root'], d['test_domain'])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    model = build_model(cfg.get('model', 'adman'), cfg['training']['class_num'], source_count=2).to(device)
    state = checkpoint_state_dict(model_path, device)
    model.load_state_dict(state, strict=False); model.eval()
    ys, ps, probs, p1s, p2s = [], [], [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device); a, b = model(x); a, b = F.softmax(a, 1), F.softmax(b, 1)
            # ``Tensor.max(dim)`` returns (values, indices); only values are
            # needed as confidence weights for fusing the two ADMAN heads.
            c1, c2 = a.max(dim=1).values, b.max(dim=1).values
            total = (c1 + c2).clamp_min(1e-12)
            fused = c1.div(total).unsqueeze(1) * a + c2.div(total).unsqueeze(1) * b
            ys += y.cpu().tolist(); ps += fused.argmax(1).cpu().tolist(); probs += fused.cpu().tolist()
            p1s += a.argmax(1).cpu().tolist(); p2s += b.argmax(1).cpu().tolist()
    names = ['AD', 'NC']
    accuracy = 100. * sum(a == b for a, b in zip(ys, ps)) / max(len(ys), 1)
    paths = [Path(p).name for p in dataset.file_paths]
    out = Path(out_dir or d['output_dir']); tag = tag or Path(model_path).stem
    out = out / tag
    tag = None
    summary = save_classification_artifacts(
        out, tag, paths, ys, ps, probabilities=probs,
        extra_columns={'branch1_pred': p1s, 'branch2_pred': p2s},
        class_names=names,
        metadata={'method': cfg.get('model', 'adman'), 'accuracy_percent': accuracy,
                  'checkpoint': str(model_path)},
    )
    result = {'accuracy': accuracy, 'auc': summary.get('roc_auc'),
              'report': summary['classification_report'], 'cm': summary['confusion_matrix'],
              'ys': ys, 'ps': ps, 'probs': probs, 'p1s': p1s, 'p2s': p2s, 'paths': paths,
              'metrics': summary}
    print('{} accuracy: {:.2f}% ({}/{})'.format(
        cfg.get('model', 'adman'), accuracy, sum(a == b for a, b in zip(ys, ps)), len(ys)))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', default='configs/eval/adman/adni.yaml'); parser.add_argument('--model-path', dest='model_path'); parser.add_argument('--tag'); parser.add_argument('--out-dir'); parser.add_argument('--batch-size', type=int)
    args = parser.parse_args(); evaluate(load_config(args.config), args.model_path, args.tag, args.out_dir, args.batch_size)
