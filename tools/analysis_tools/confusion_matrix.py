# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import tempfile

import torch
import numpy as np
from mmengine import dump, list_from_file, load
from mmengine.config import Config, DictAction
from mmengine.evaluator import Evaluator
from mmengine.runner import Runner

from mmaction.evaluation import ConfusionMatrix
from mmaction.registry import DATASETS
from mmaction.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(
        description='Eval a checkpoint and draw the confusion matrix.')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'ckpt_or_result',
        type=str,
        help='The checkpoint file (.pth) or '
             'dumpped predictions pickle file (.pkl).')
    parser.add_argument('--out', help='the file to save the confusion matrix.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to display the metric result by matplotlib if supports.')
    parser.add_argument(
        '--show-path', type=str, help='Path to save the visualization image.')
    parser.add_argument(
        '--include-values',
        action='store_true',
        help='To draw the values in the figure.')
    parser.add_argument('--label-file', default=None, help='Labelmap file')
    parser.add_argument(
        '--target-classes',
        type=int,
        nargs='+',
        default=[],
        help='Selected classes to evaluate, and remains will be neglected')
    parser.add_argument(
        '--cmap',
        type=str,
        default='viridis',
        help='The color map to use. Defaults to "viridis".')
    parser.add_argument(
        '--font-size',
        type=int,
        default=12,
        help='Font size for all text in the confusion matrix.')
    parser.add_argument(
        '--decimal-places',
        type=int,
        default=2,
        help='Number of decimal places for decimal values (0-3).')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    args = parser.parse_args()
    return args


def plot_confusion_matrix_with_decimal(cm, classes, include_values=True,
                                       cmap='viridis', fontsize=12,
                                       decimal_places=2, show=True):
    """Plot confusion matrix with decimal values (0-1)."""
    import matplotlib.pyplot as plt

    # Convert to numpy array if it's a tensor
    if isinstance(cm, torch.Tensor):
        cm_np = cm.cpu().numpy()
    else:
        cm_np = cm

    # Calculate row-wise decimal values (prediction distribution for each true class)
    row_sums = cm_np.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    cm_decimal = cm_np.astype('float') / row_sums

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_decimal, interpolation='nearest', cmap=cmap)

    # Add colorbar with consistent font size
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=fontsize)

    # Set all text with the same fontsize
    ax.set(xticks=np.arange(cm_decimal.shape[1]),
           yticks=np.arange(cm_decimal.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Set font size for tick labels
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize)

    # Set axis label font size (same as other text)
    ax.set_xlabel('Predicted label', fontsize=fontsize)
    ax.set_ylabel('True label', fontsize=fontsize)

    # Set title with the same font size
    ax.set_title("Confusion Matrix (Row Normalized)", fontsize=fontsize, pad=20)

    # Loop over data dimensions and create text annotations with decimal values
    if include_values:
        # Create format string based on decimal places
        if decimal_places == 0:
            fmt = '.0f'
        elif decimal_places == 1:
            fmt = '.1f'
        elif decimal_places == 2:
            fmt = '.2f'
        else:  # decimal_places >= 3
            fmt = '.3f'

        thresh = cm_decimal.max() / 2.
        for i in range(cm_decimal.shape[0]):
            for j in range(cm_decimal.shape[1]):
                value = cm_decimal[i, j]
                # Display only decimal value
                text = f'{value:{fmt}}'
                ax.text(j, i, text,
                        ha="center", va="center",
                        color="white" if cm_decimal[i, j] > thresh else "black",
                        fontsize=fontsize)

    fig.tight_layout()
    return fig


def main():
    args = parse_args()

    # register all modules in mmaction into the registries
    # do not init the default scope here because it will be init in the runner
    register_all_modules(init_default_scope=False)

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.ckpt_or_result.endswith('.pth'):
        # Set confusion matrix as the metric.
        cfg.test_evaluator = dict(type='ConfusionMatrix')

        cfg.load_from = str(args.ckpt_or_result)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.work_dir = tmpdir
            runner = Runner.from_cfg(cfg)
            classes = runner.test_loop.dataloader.dataset.metainfo.get(
                'classes')
            cm = runner.test()['confusion_matrix/result']
            logging.shutdown()
    else:
        predictions = load(args.ckpt_or_result)
        evaluator = Evaluator(ConfusionMatrix())
        metrics = evaluator.offline_evaluate(predictions, None)
        cm = metrics['confusion_matrix/result']
        try:
            # Try to build the dataset.
            dataset = DATASETS.build({
                **cfg.test_dataloader.dataset, 'pipeline': []
            })
            classes = dataset.metainfo.get('classes')
        except Exception:
            classes = None

    if args.label_file is not None:
        classes = list_from_file(args.label_file)
    if classes is None:
        num_classes = cm.shape[0]
        classes = list(range(num_classes))

    if args.target_classes:
        assert len(args.target_classes) > 1, \
            'please ensure select more than one class'
        target_idx = torch.tensor(args.target_classes)
        cm = cm[target_idx][:, target_idx]
        classes = [classes[idx] for idx in target_idx]

    if args.out is not None:
        dump(cm, args.out)

    if args.show or args.show_path is not None:
        # Use our custom plotting function instead of ConfusionMatrix.plot
        fig = plot_confusion_matrix_with_decimal(
            cm,
            show=args.show,
            classes=classes,
            include_values=args.include_values,
            cmap=args.cmap,
            fontsize=args.font_size,
            decimal_places=args.decimal_places)

        if args.show_path is not None:
            fig.savefig(args.show_path, dpi=300, bbox_inches='tight')
            print(f'The confusion matrix is saved at {args.show_path}.')


if __name__ == '__main__':
    main()