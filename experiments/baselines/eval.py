"""Evaluation for a saved DANN, DSAN or DeepCORAL state dict."""
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models import build_model
from tools.data_loader import NiiDataset
from utils.checkpoint import load_checkpoint
from utils.config import get_device, load_config
from utils.reporting import save_classification_artifacts

def evaluate(cfg, method=None, model_path=None):
    method = str(method or cfg.get('model', 'dann')).lower(); device = get_device(cfg); model_path = model_path or cfg['checkpoint']['model_path']
    model = build_model(method, cfg['training']['class_num'], source_count=1).to(device)
    incompatible, mapping, loaded_count, model_count = load_checkpoint(model, model_path, device, strict=False)
    print('Loaded checkpoint: mapping={}, matched={}/{}'.format(mapping, loaded_count, model_count))
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print('checkpoint key mismatch: missing={}, unexpected={}'.format(len(incompatible.missing_keys), len(incompatible.unexpected_keys)))
    model.eval()
    ds = NiiDataset(cfg['evaluation']['test_root'], cfg['evaluation']['test_domain'])
    loader = DataLoader(ds, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=0)
    good = 0; ys = []; preds = []; probabilities = []
    names = [Path(p).name for p in ds.file_paths]
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model.predict(x) if method == 'dsan' else model(x, x)[0]
            prob = F.softmax(logits, dim=1)
            batch_pred = prob.argmax(1)
            good += batch_pred.eq(y).sum().item()
            ys.extend(y.cpu().tolist()); preds.extend(batch_pred.cpu().tolist())
            probabilities.extend(prob.cpu().tolist())
    acc = 100.0 * good / max(len(ds), 1)
    run_name = cfg.get('evaluation', {}).get('run_name') or Path(model_path).stem
    out = Path(cfg.get('evaluation', {}).get('output_dir', 'eval_results')) / run_name
    class_num = int(cfg.get('training', {}).get('class_num', 2))
    class_names = ['AD', 'NC'] if class_num == 2 else [str(i) for i in range(class_num)]
    metrics = save_classification_artifacts(
        out, None, names, ys, preds, probabilities=probabilities,
        class_names=class_names, metadata={'method': method, 'accuracy_percent': acc,
                                           'checkpoint': str(model_path)},
    )
    print('{} accuracy: {:.2f}% ({}/{})'.format(method, acc, good, len(ds)))
    return acc

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--method', choices=['dann', 'dsan', 'deepcoral', 'signal_adman']); p.add_argument('--model-path'); a = p.parse_args(); evaluate(load_config(a.config), a.method, a.model_path)
