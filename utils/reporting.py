"""Shared metric and artifact persistence helpers for experiments."""
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _artifact_name(kind, tag, suffix):
    return '{}_{}{}'.format(kind, tag, suffix) if tag else '{}{}'.format(kind, suffix)


def classification_metrics(y_true, y_pred, probabilities=None, class_names=None):
    """Return JSON-compatible accuracy, AUC, confusion matrix and full report."""
    y_true = list(map(int, y_true)); y_pred = list(map(int, y_pred))
    n_classes = len(class_names) if class_names else max(y_true + y_pred + [1]) + 1
    labels = list(range(n_classes)); names = class_names or [str(x) for x in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, target_names=names,
                                   output_dict=True, zero_division=0)
    correct = sum(a == b for a, b in zip(y_true, y_pred))
    result = {
        'class_names': names,
        'count': len(y_true),
        'correct': correct,
        'accuracy': float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        'accuracy_percent': 100.0 * correct / max(len(y_true), 1),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)) if y_true else 0.0,
        'precision_macro': float(precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
        'precision_weighted': float(precision_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
        'recall_weighted': float(recall_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)),
        'f1_macro': float(f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
    }
    if probabilities is not None and n_classes == 2:
        try:
            result['roc_auc'] = float(roc_auc_score(y_true, [float(p[1]) for p in probabilities]))
        except ValueError:
            result['roc_auc'] = None
    return result


def metric_log_fields(metrics):
    """Select scalar classification metrics for one training-log row."""
    keys = (
        'accuracy_percent', 'balanced_accuracy', 'precision_macro',
        'precision_weighted', 'recall_macro', 'recall_weighted',
        'f1_macro', 'f1_weighted', 'roc_auc', 'correct', 'count',
    )
    return {key: metrics.get(key) for key in keys}


def save_metrics(output_dir, tag, metrics, y_true=None, y_pred=None, probabilities=None,
                 file_names=None):
    """Persist metrics JSON, confusion matrix CSV, optional predictions CSV and report text."""
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    stem = str(tag) if tag else ''
    (output / _artifact_name('metrics', stem, '.json')).write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=_jsonable), encoding='utf-8')
    cm = metrics.get('confusion_matrix', [])
    with (output / _artifact_name('confusion_matrix', stem, '.csv')).open('w', newline='') as handle:
        class_names = metrics.get('class_names', list(range(len(cm))))
        writer = csv.writer(handle); writer.writerow(['true\\pred'] + class_names)
        for name, row in zip(class_names, cm): writer.writerow([name] + row)
    (output / _artifact_name('confusion_matrix', stem, '.json')).write_text(
        json.dumps({'labels': metrics.get('class_names', []), 'matrix': cm}, indent=2), encoding='utf-8')
    report = metrics.get('classification_report', {})
    if y_true is not None and y_pred is not None:
        labels = list(range(len(metrics.get('class_names', []))))
        report_text = classification_report(
            y_true, y_pred, labels=labels, target_names=metrics.get('class_names'),
            digits=4, zero_division=0)
    else:
        report_text = json.dumps(report, indent=2, ensure_ascii=False)
    (output / _artifact_name('classification_report', stem, '.txt')).write_text(
        report_text, encoding='utf-8')
    (output / _artifact_name('classification_report', stem, '.json')).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    if y_true is not None and y_pred is not None:
        names = list(file_names or [str(i) for i in range(len(y_true))])
        with (output / _artifact_name('predictions', stem, '.csv')).open('w', newline='') as handle:
            probability_columns = ['prob_' + str(name) for name in metrics.get('class_names', [])]
            writer = csv.writer(handle); writer.writerow(['file_name', 'true_label', 'pred_label'] + probability_columns)
            for i, (name, true, pred) in enumerate(zip(names, y_true, y_pred)):
                prob = probabilities[i] if probabilities is not None and i < len(probabilities) else ''
                writer.writerow([name, true, pred] + (list(prob) if prob != '' else [''] * len(probability_columns)))


def append_training_log(output_dir, row):
    """Append one training/evaluation row to output_dir/training_log.csv."""
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    path = output / 'training_log.csv'; row = {str(k): _jsonable(v) for k, v in row.items()}
    exists = path.exists()
    with path.open('a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists: writer.writeheader()
        writer.writerow(row)
    message = ' '.join('{}={}'.format(key, value) for key, value in row.items())
    with (output / 'training.log').open('a', encoding='utf-8') as handle:
        handle.write('[{}] {}\n'.format(datetime.now().isoformat(timespec='seconds'), message))


def append_text_log(output_dir, message):
    """Print and append one human-readable training message."""
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    print(message, flush=True)
    with (output / 'training.log').open('a', encoding='utf-8') as handle:
        handle.write(str(message) + '\n')


def save_classification_artifacts(output_dir, tag, file_names, y_true, y_pred,
                                  probabilities=None, extra_columns=None,
                                  class_names=None, metadata=None):
    """Write a complete, consistent set of classification evaluation files."""
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    tag = str(tag) if tag else ''; y_true = list(map(int, y_true)); y_pred = list(map(int, y_pred))
    file_names = [str(name) for name in file_names]
    if not (len(file_names) == len(y_true) == len(y_pred)):
        raise ValueError('file_names, y_true and y_pred must have equal lengths')
    n_classes = len(class_names) if class_names else max(y_true + y_pred + [1]) + 1
    labels = list(range(n_classes)); names = list(class_names or map(str, labels))
    probabilities = ([[float(v) for v in row] for row in probabilities]
                     if probabilities is not None else None)
    extras = extra_columns or {}
    if probabilities is not None and len(probabilities) != len(y_true):
        raise ValueError('probabilities must have one row per sample')
    for key, values in extras.items():
        if len(values) != len(y_true):
            raise ValueError('extra column %r must have one value per sample' % key)

    columns = ['file_name', 'true_label', 'pred_label']
    if probabilities:
        columns.extend('prob_%s' % name for name in names[:len(probabilities[0])])
    columns.extend(extras)
    with (output / _artifact_name('predictions', tag, '.csv')).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle); writer.writerow(columns)
        for i, (file_name, true, pred) in enumerate(zip(file_names, y_true, y_pred)):
            row = [file_name, true, pred]
            if probabilities: row.extend(probabilities[i])
            row.extend(extras[key][i] for key in extras); writer.writerow(row)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, target_names=names,
                                   output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, labels=labels, target_names=names,
                                        digits=4, zero_division=0)
    metrics = {
        'tag': tag, 'count': len(y_true), 'class_names': names,
        'accuracy': float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)) if y_true else 0.0,
        'precision_macro': float(precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
        'precision_weighted': float(precision_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
        'recall_weighted': float(recall_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)),
        'f1_macro': float(f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)),
        'confusion_matrix': cm.tolist(), 'classification_report': report,
    }
    if metadata: metrics.update(metadata)
    if probabilities and n_classes == 2 and len(probabilities[0]) > 1 and len(set(y_true)) > 1:
        try: metrics['roc_auc'] = float(roc_auc_score(y_true, [row[1] for row in probabilities]))
        except ValueError: metrics['roc_auc'] = None

    with (output / _artifact_name('confusion_matrix', tag, '.csv')).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle); writer.writerow(['true\\pred'] + names)
        for name, row in zip(names, cm.tolist()): writer.writerow([name] + row)
    (output / _artifact_name('confusion_matrix', tag, '.json')).write_text(
        json.dumps({'labels': names, 'matrix': cm.tolist()}, indent=2), encoding='utf-8')
    (output / _artifact_name('classification_report', tag, '.json')).write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    (output / _artifact_name('classification_report', tag, '.txt')).write_text(report_text, encoding='utf-8')
    (output / _artifact_name('metrics', tag, '.json')).write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=_jsonable), encoding='utf-8')
    return metrics
