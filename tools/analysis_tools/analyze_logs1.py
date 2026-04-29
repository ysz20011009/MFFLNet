import argparse
import json
from collections import defaultdict
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def cal_train_time(log_dicts, args):
    """计算训练时间统计"""
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
    """绘制训练曲线"""
    if args.backend is not None:
        plt.switch_backend(args.backend)
    sns.set_style(args.style)

    # 设置画布大小
    plt.figure(figsize=(12, 7))

    # 生成图例名称
    legend_names = []
    if args.legend and len(args.legend) == len(args.json_logs):
        # 使用用户提供的图例名称
        legend_names = args.legend
    else:
        # 使用文件名作为图例名称
        legend_names = [os.path.basename(log).replace('.json', '')
                        for log in args.json_logs]

    # 定义颜色主题（每个JSON文件分配一个主色）
    color_themes = [
        {'main': '#1f77b4', 'light': '#aec7e8'},  # 蓝色系
        {'main': '#ff7f0e', 'light': '#ffbb78'},  # 橙色系
        {'main': '#2ca02c', 'light': '#98df8a'},  # 绿色系
        {'main': '#d62728', 'light': '#ff9896'},  # 红色系
        {'main': '#9467bd', 'light': '#c5b0d5'},  # 紫色系
        {'main': '#8c564b', 'light': '#c49c94'},  # 棕色系
        {'main': '#e377c2', 'light': '#f7b6d2'},  # 粉色系
        {'main': '#7f7f7f', 'light': '#c7c7c7'},  # 灰色系
    ]

    # 定义线型和标记
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    # 创建左右坐标轴
    ax1 = plt.gca()  # 左轴 - loss
    ax2 = ax1.twinx()  # 右轴 - accuracy

    all_lines = []  # 收集所有线条用于图例
    all_labels = []  # 收集所有标签

    for i, (log_dict, legend_name) in enumerate(zip(log_dicts, legend_names)):
        if not log_dict:
            print(f"警告: {args.json_logs[i]} 没有有效数据!")
            continue

        epochs = sorted(log_dict.keys())

        # 为每个JSON文件分配颜色主题
        color_theme = color_themes[i % len(color_themes)]
        line_style = line_styles[i % len(line_styles)]
        marker = markers[i % len(markers)]

        # 收集数据
        epoch_list = []
        avg_losses = []
        acc_top1_list = []

        for epoch in epochs:
            # 收集loss数据
            if 'loss' in log_dict[epoch]:
                loss_values = log_dict[epoch]['loss']
                if loss_values:
                    avg_loss = np.mean(loss_values)
                    epoch_list.append(epoch)
                    avg_losses.append(avg_loss)

            # 收集acc数据
            if 'acc/top1' in log_dict[epoch]:
                acc_values = log_dict[epoch]['acc/top1']
                if acc_values:
                    acc_top1_list.append(acc_values[-1])

        # 确保数据对齐
        if len(epoch_list) > len(acc_top1_list):
            epoch_list = epoch_list[:len(acc_top1_list)]

        # 绘制loss曲线（左轴，深色，主线条）
        if avg_losses:
            line1 = ax1.plot(epoch_list, avg_losses,
                             color=color_theme['main'],
                             linewidth=2,
                             linestyle=line_style,
                             marker=marker,
                             markersize=6,
                             markerfacecolor='white',
                             markeredgewidth=1.5,
                             markeredgecolor=color_theme['main'],
                             label=f'{legend_name} - Loss')
            all_lines.append(line1[0])
            all_labels.append(f'{legend_name} - Loss')

        # 绘制accuracy曲线（右轴，浅色，细线条）
        if acc_top1_list and epoch_list:
            line2 = ax2.plot(epoch_list, acc_top1_list,
                             color=color_theme['light'],
                             linewidth=1.5,
                             linestyle=line_style,
                             marker=marker,
                             markersize=5,
                             alpha=0.8,
                             label=f'{legend_name} - Acc/Top1')
            all_lines.append(line2[0])
            all_labels.append(f'{legend_name} - Acc/Top1')

    # 设置坐标轴标签和样式
    ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Average Loss', fontsize=14, fontweight='bold', color='#333333')
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12, labelcolor='#333333')

    # 去除网格线（根据需求修改）
    ax1.grid(False)  # 设置为False以去除网格线

    ax2.set_ylabel('Top1 Accuracy (%)', fontsize=14, fontweight='bold', color='#666666')
    ax2.tick_params(axis='y', labelsize=12, labelcolor='#666666')

    # 设置标题
    if args.title:
        plt.title(args.title, fontsize=16, fontweight='bold', pad=20)
    else:
        plt.title('Training Loss and Top1 Accuracy Curves',
                  fontsize=16, fontweight='bold', pad=20)

    # 添加图例
    if all_lines:
        ax1.legend(all_lines, all_labels,
                   loc='upper center',
                   bbox_to_anchor=(0.5, -0.15),
                   ncol=min(4, len(legend_names)),
                   fontsize=11,
                   frameon=True,
                   framealpha=0.9,
                   fancybox=True,
                   shadow=False)

    # 调整布局
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # 为底部图例留出空间

    # 保存或显示
    if args.out:
        print(f'保存图表到: {args.out}')
        plt.savefig(args.out, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def add_plot_parser(subparsers):
    parser_plt = subparsers.add_parser(
        'plot_curve',
        help='绘制训练曲线 (loss和accuracy)'
    )
    parser_plt.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='JSON格式的训练日志文件路径 (支持多个文件)'
    )
    parser_plt.add_argument(
        '--title',
        type=str,
        help='图表标题'
    )
    parser_plt.add_argument(
        '--legend',
        type=str,
        nargs='+',
        default=None,
        help='每条曲线的图例名称 (数量需与json_logs一致)'
    )
    parser_plt.add_argument(
        '--backend',
        type=str,
        default=None,
        help='matplotlib后端 (如: TkAgg, Qt5Agg, Agg)'
    )
    parser_plt.add_argument(
        '--style',
        type=str,
        default='white',  # 改为white样式，默认无网格
        help='seaborn样式 (如: white, darkgrid, whitegrid, dark)'
    )
    parser_plt.add_argument(
        '--out',
        type=str,
        default=None,
        help='输出图片文件路径 (如: ./result.png)'
    )
    parser_plt.add_argument(
        '--grid',
        action='store_true',
        default=False,
        help='显示网格线 (默认不显示)'
    )


def add_time_parser(subparsers):
    parser_time = subparsers.add_parser(
        'cal_train_time',
        help='计算平均训练时间'
    )
    parser_time.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='JSON格式的训练日志文件路径'
    )
    parser_time.add_argument(
        '--include-outliers',
        action='store_true',
        help='计算时包含每个epoch的第一个值'
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='分析JSON训练日志工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python analyze_logs.py plot_curve log1.json log2.json --legend "模型A" "模型B" --title "模型对比"
  python analyze_logs.py plot_curve log1.json --out ./result.png
  python analyze_logs.py cal_train_time log1.json log2.json
        """
    )
    subparsers = parser.add_subparsers(dest='task', help='任务选择')
    add_plot_parser(subparsers)
    add_time_parser(subparsers)
    args = parser.parse_args()

    # 验证图例参数
    if hasattr(args, 'legend') and args.legend:
        if len(args.legend) != len(args.json_logs):
            parser.error(f'--legend参数数量 ({len(args.legend)}) 必须与json_logs数量 ({len(args.json_logs)}) 一致')

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
                content = re.sub(r'\}\s*\{', '},{', content)
                if not content.strip().startswith('['):
                    content = '[' + content + ']'
                content = re.sub(r',\s*]', ']', content)
                content = re.sub(r',\s*}', '}', content)

                try:
                    logs = json.loads(content)
                except json.JSONDecodeError:
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

                print(f"  找到 {len(logs)} 条日志记录")

                for log in logs:
                    # 处理acc/top1记录（这些记录没有epoch，需要特殊处理）
                    if 'acc/top1' in log and 'epoch' not in log:
                        epoch = log.get('step', 0)
                        log['epoch'] = epoch

                    if 'epoch' not in log:
                        continue

                    epoch = log['epoch']
                    for key, value in log.items():
                        if key == 'epoch':
                            continue
                        if not isinstance(value, (list, tuple)):
                            value = [value]
                        log_dicts[idx][epoch][key].extend(value)

        except Exception as e:
            print(f"加载日志文件 {json_log} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    # 打印统计信息
    for i, log_dict in enumerate(log_dicts):
        print(f"\n日志文件 {os.path.basename(json_logs[i])} 统计:")
        if not log_dict:
            print("  空日志")
            continue

        epochs = sorted(log_dict.keys())
        print(f"  包含 {len(epochs)} 个epoch")

        if epochs:
            metrics = log_dict[epochs[0]].keys()
            print(f"  包含指标: {', '.join(metrics)}")

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