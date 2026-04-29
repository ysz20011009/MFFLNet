# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
from collections import defaultdict
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

def batch_process(input_dir):
    results = []
    files = sorted(
        [f for f in os.listdir(input_dir) if f.endswith('.pkl')],
        key=lambda x: int(re.search(r'\d+', x).group())
    )
    for filename in files:#os.listdir(input_dir)
        if filename.endswith('.pkl'):
            file_path = os.path.join(input_dir, filename)
            try:
                with open(file_path, 'rb') as f:
                    loaded = pickle.load(f)
                # print(loaded)
                video_preds = []
                video_labels = []
                for i in loaded:
                    video_preds.append(i['pred_score'].tolist())
                    video_labels.append(i['gt_label'].item())

                    # break
                #
                video_labels = np.array(video_labels)
                video_preds = np.array(video_preds)
                #video_preds = video_preds[:, 1]
                # print(len(video_labels))
                # print(video_preds[0])
                # print(len(video_preds[0]))
                # video_preds = loaded['video_preds']
                # video_labels = loaded['video_labels']
                # # print(np.array(video_preds))
                # print(np.array(video_labels))
                AP = average_precision_score(video_labels, video_preds[:, 1])
                results.append(AP)
            except Exception as e:
                print(f"Error processing{filename}:{str(e)}")
        #print(f"name is:{filename}")
    return results
APs = batch_process('/nfs/ysz/uniformerv2_result/MEDIA/dongjie/mona/results')


def cal_train_time(log_dicts, args):
    for i, log_dict in enumerate(log_dicts):
        print(f'{"-" * 5}Analyze train time of {args.json_logs[i]}{"-" * 5}')
        all_times = []
        for epoch in log_dict.keys():
            if args.include_outliers:
                all_times.append(log_dict[epoch]['time'])
            else:
                all_times.append(log_dict[epoch]['time'][1:])
        all_times = np.array(all_times)
        epoch_ave_time = all_times.mean(-1)
        slowest_epoch = epoch_ave_time.argmax()
        fastest_epoch = epoch_ave_time.argmin()
        std_over_epoch = epoch_ave_time.std()
        print(f'slowest epoch {slowest_epoch + 1}, '
              f'average time is {epoch_ave_time[slowest_epoch]:.4f}')
        print(f'fastest epoch {fastest_epoch + 1}, '
              f'average time is {epoch_ave_time[fastest_epoch]:.4f}')
        print(f'time std over epochs is {std_over_epoch:.4f}')
        print(f'average iter time: {np.mean(all_times):.4f} s/iter')
        print()


def plot_curve(log_dicts, args):
    if args.backend is not None:
        plt.switch_backend(args.backend)
    sns.set_style(args.style)

    # 使用文件名作为图例前缀
    legend_prefix = [os.path.basename(log).replace('.json', '') for log in args.json_logs]

    # 自动生成图例
    legend = args.legend
    if legend is None:
        legend = []
        for i, json_log in enumerate(args.json_logs):
            for metric in args.keys:
                legend.append(f'{legend_prefix[i]}_{metric}')

    metrics = args.keys
    num_metrics = len(metrics)

    # 设置图表尺寸和布局
    plt.figure(figsize=(10, 6))

    for i, log_dict in enumerate(log_dicts):
        # 获取所有epoch并按顺序排序
        epochs = sorted(log_dict.keys())
        if not epochs:
            print(f"警告: {args.json_logs[i]} 没有有效数据!")
            continue

        for j, metric in enumerate(metrics):
            print(f'绘制曲线: {args.json_logs[i]}, 指标: {metric}')

            # 存储每个epoch的平均值
            epoch_avgs = []
            valid_epochs = []

            for epoch in epochs:
                # 获取该epoch的所有指标值
                metric_values = log_dict[epoch].get(metric, [])

                if not metric_values:
                    print(f"  跳过epoch {epoch}，无{metric}数据")
                    continue

                # 计算该epoch的平均值
                avg_value = np.mean(metric_values)
                epoch_avgs.append(avg_value)
                valid_epochs.append(epoch)

                print(f"  Epoch {epoch}: {metric} 平均值 = {avg_value:.6f}")

            if not valid_epochs:
                print(f"警告: {args.json_logs[i]} 的 {metric} 没有足够数据绘图")
                continue
            plt.plot()
            # 绘制曲线
            plt.xlabel('Epoch')
            plt.ylabel('Value')


            # 为每个指标使用不同的线型
            linestyles = ['-', '--', '-.', ':']
            linestyle = linestyles[j % len(linestyles)]
            plt.plot(valid_epochs,APs,label='APs',color='red',marker='v',markersize=6)
            plt.plot(
                valid_epochs,
                epoch_avgs,
                label='Loss',  # legend[i * num_metrics + j]
                linewidth=2,
                linestyle=linestyle,
                marker='o',  # 在每个epoch点上添加标记
                markersize=6
            )

        if args.title is not None:
            plt.title(args.title)

    # 添加图例和网格
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 设置x轴为整数刻度
    all_epochs = []
    for log_dict in log_dicts:
        all_epochs.extend(log_dict.keys())

    if all_epochs:
        min_epoch = min(all_epochs)
        max_epoch = max(all_epochs)
        plt.xticks(np.arange(min_epoch, max_epoch + 1, max(1, (max_epoch - min_epoch) // 10)))

    if args.out is None:
        plt.show()
    else:
        print(f'保存图表到: {args.out}')
        plt.savefig(args.out, dpi=300, bbox_inches='tight')
        plt.cla()


def add_plot_parser(subparsers):
    parser_plt = subparsers.add_parser(
        'plot_curve', help='绘制训练曲线')
    parser_plt.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='JSON格式的训练日志文件路径')
    parser_plt.add_argument(
        '--keys',
        type=str,
        nargs='+',
        default=['loss'],
        help='要绘制的指标名称')
    parser_plt.add_argument('--title', type=str, help='图表标题')
    parser_plt.add_argument(
        '--legend',
        type=str,
        nargs='+',
        default=None,
        help='每条曲线的图例名称')
    parser_plt.add_argument(
        '--backend', type=str, default=None, help='matplotlib后端')
    parser_plt.add_argument(
        '--style', type=str, default='whitegrid', help='图表样式')
    parser_plt.add_argument('--out', type=str, default=None, help='输出文件路径')


def add_time_parser(subparsers):
    parser_time = subparsers.add_parser(
        'cal_train_time',
        help='计算平均训练时间')
    parser_time.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='JSON格式的训练日志文件路径')
    parser_time.add_argument(
        '--include-outliers',
        action='store_true',
        help='计算时包含每个epoch的第一个值')


def parse_args():
    parser = argparse.ArgumentParser(description='分析JSON日志')
    subparsers = parser.add_subparsers(dest='task', help='任务选择')
    add_plot_parser(subparsers)
    add_time_parser(subparsers)
    args = parser.parse_args()
    return args


def load_json_logs(json_logs):
    """加载并解析JSON格式的训练日志"""
    log_dicts = [defaultdict(lambda: defaultdict(list)) for _ in json_logs]

    for idx, json_log in enumerate(json_logs):
        print(f"加载日志文件: {json_log}")
        try:
            with open(json_log, 'r') as log_file:
                content = log_file.read()

                # 修复常见的JSON格式问题
                # 1. 移除多余的大括号
                content = re.sub(r'\}\s*\{', '},{', content)
                # 2. 添加缺失的方括号
                if not content.strip().startswith('['):
                    content = '[' + content + ']'
                # 3. 修复多余的逗号
                content = re.sub(r',\s*]', ']', content)
                content = re.sub(r',\s*}', '}', content)

                # 尝试解析为JSON数组
                try:
                    logs = json.loads(content)
                except json.JSONDecodeError:
                    # 如果整体解析失败，尝试逐行解析
                    logs = []
                    for line in content.splitlines():
                        line = line.strip()
                        if not line or line in ['[', ']']:
                            continue
                        if line.endswith(','):
                            line = line[:-1]
                        try:
                            log = json.loads(line)
                            logs.append(log)
                        except json.JSONDecodeError:
                            print(f"跳过无法解析的行: {line[:100]}...")

                print(f"找到 {len(logs)} 条日志记录")

                for log in logs:
                    # 确保有epoch字段
                    if 'epoch' not in log:
                        print(f"跳过没有epoch字段的记录: {log}")
                        continue

                    epoch = log['epoch']
                    # 处理所有指标
                    for key, value in log.items():
                        if key == 'epoch':
                            continue
                        # 确保值是可迭代的
                        if not isinstance(value, (list, tuple)):
                            value = [value]
                        log_dicts[idx][epoch][key].extend(value)

        except Exception as e:
            print(f"加载日志文件 {json_log} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    # 打印调试信息
    for i, log_dict in enumerate(log_dicts):
        print(f"\n日志文件 {json_logs[i]} 结构:")
        if not log_dict:
            print("  空日志")
            continue

        epochs = sorted(log_dict.keys())
        print(f"  包含 {len(epochs)} 个epoch")

        for epoch in epochs:
            print(f"  Epoch {epoch}:")
            metrics = log_dict[epoch]
            for metric, values in metrics.items():
                avg_value = np.mean(values) if values else "无数据"
                print(f"    {metric}: {len(values)} 个值, 平均值 = {avg_value}")

    return log_dicts


def main():
    args = parse_args()

    json_logs = args.json_logs
    for json_log in json_logs:
        if not os.path.exists(json_log):
            raise FileNotFoundError(f"日志文件不存在: {json_log}")
        if not json_log.endswith('.json'):
            print(f"警告: 文件 {json_log} 不是JSON扩展名，但仍尝试处理")

    log_dicts = load_json_logs(json_logs)

    if hasattr(args, 'task') and args.task:
        eval(args.task)(log_dicts, args)
    else:
        print("错误: 未指定任务。可用任务: plot_curve, cal_train_time")


if __name__ == '__main__':
    main()