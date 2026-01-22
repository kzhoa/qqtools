#!/usr/bin/env python3
"""
高斯Log文件处理器
从高斯log文件中提取关键信息并保存为JSON格式
支持处理单个文件或整个目录
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

file_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(file_dir))

# 导入您的模块
try:
    from gaus_reader import create_g16_reader

    READER_INSTANCE = create_g16_reader(opt=False)  # for single-point log files without structure optimization
except ImportError as e:
    print("错误：无法从 gaus_reader 模块导入 create_g16_reader。请确保gaus_reader.py在Python路径中。")
    print(e)
    sys.exit(1)


def convert_numpy_types(obj):
    """
    递归地将numpy数据类型转换为Python原生类型，使其可JSON序列化
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.complex64, np.complex128)):
        return complex(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.bytes_):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


def process_single_log_file(input_file, output_file=None, verbose=False):
    """
    处理单个高斯log文件并提取信息

    Args:
        input_file (Path): 输入的log文件路径(Path对象)
        output_file (Path, optional): 输出的JSON文件路径(Path对象)
        verbose (bool): 是否显示详细信息

    Returns:
        dict: 提取的结果数据
    """
    if verbose:
        print(f"正在处理文件: {input_file}")

    # 处理文件
    try:
        results = READER_INSTANCE.read_file(str(input_file))

        results = convert_numpy_types(results)
        if verbose:
            print(f"成功提取 {len(results)} 个数据项")

    except Exception as e:
        print(f"处理文件 {input_file} 时出错: {e}")
        traceback.print_exc()
        return None

    # 确定输出文件路径
    if output_file is None:
        # 自动生成输出文件名（与输入文件同名，扩展名为.json）
        output_path = input_file.parent / (input_file.stem + ".json")
        if verbose:
            print(f"未指定输出文件，将保存到: {output_path}")
    else:
        output_path = output_file
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        if verbose:
            print(f"结果已保存到: {output_path}")
            print(f"文件大小: {output_path.stat().st_size} bytes")

    except Exception as e:
        print(f"保存结果到 {output_path} 时出错: {e}")
        raise e

    return results


def process_directory(input_dir, output_base_dir=None, verbose=False):
    """
    处理目录下的所有log文件

    Args:
        input_dir (Path): 输入的目录路径
        output_base_dir (Path, optional): 输出的基础目录路径
        verbose (bool): 是否显示详细信息

    Returns:
        dict: 处理结果统计信息
    """
    if verbose:
        print(f"正在扫描目录: {input_dir} 中的log文件")

    # 递归查找所有.log文件
    log_files = list(input_dir.rglob("*.log"))

    if not log_files:
        print(f"在目录 {input_dir} 中未找到任何.log或.out文件")
        return {"total_files": 0, "processed_files": 0, "failed_files": 0}

    if verbose:
        print(f"找到 {len(log_files)} 个log/out文件")

    stats = {
        "total_files": len(log_files),
        "processed_files": 0,
        "failed_files": 0,
        "failed_file_list": [],  # 记录失败文件列表
    }

    for log_file in log_files:
        if verbose:
            print(f"\n处理文件 {stats['processed_files'] + 1}/{stats['total_files']}: {log_file.name}")

        # 确定输出文件路径
        if output_base_dir is None:
            # 输出到与输入文件相同的目录
            output_file = log_file.parent / (log_file.stem + ".json")
        else:
            # 输出到指定的基础目录，保持相同的文件名
            output_file = output_base_dir / log_file.with_suffix(".json")
            # 确保输出目录存在
            output_file.parent.mkdir(parents=True, exist_ok=True)

        # 处理单个文件
        try:
            result = process_single_log_file(log_file, output_file, verbose)

            stats["processed_files"] += 1
        except Exception as e:
            stats["failed_files"] += 1
            # 记录失败的文件和错误信息
            stats["failed_file_list"].append(
                {
                    "file_path": str(log_file),
                    "error_message": str(e),
                }
            )

    return stats


def process_gaussian_log(input_path, output_path=None, verbose=False):
    """
    处理高斯log文件或目录并提取信息

    Args:
        input_path (str): 输入的log文件路径或目录路径
        output_path (str, optional): 输出的JSON文件路径或目录路径
        verbose (bool): 是否显示详细信息

    Returns:
        dict or None: 如果是单个文件返回结果数据，如果是目录返回统计信息
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    if input_path.is_file():
        # 处理单个文件
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # 输出到相同目录
            output_path = input_path.parent / (input_path.stem + ".json")

        return process_single_log_file(input_path, output_path, verbose)

    elif input_path.is_dir():
        # 处理目录
        if output_path is not None:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = input_path.parent  # 输出到相同目录

        return process_directory(input_path, output_path, verbose)

    else:
        raise ValueError(f"输入路径既不是文件也不是目录: {input_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="处理高斯log文件并提取关键信息为JSON格式\n支持处理单个文件或整个目录",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  单个文件:
    %(prog)s calculation.log
    %(prog)s calculation.log results.json
    %(prog)s /path/to/calculation.log /output/path/results.json

  目录处理:
    %(prog)s /path/to/log_files/
    %(prog)s /path/to/log_files/ /output/path/
    %(prog)s /path/to/log_files/ /output/path/ --verbose

  注意: 当处理目录时，如果没有指定输出路径，JSON文件将保存在与log文件相同的目录中
        """,
    )

    parser.add_argument("input_path", help="输入的高斯log文件路径或包含log文件的目录路径")
    parser.add_argument("output_path", nargs="?", default=None, help="输出的JSON文件路径或目录路径（可选）")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出模式")

    args = parser.parse_args()

    try:
        # 处理文件或目录
        result = process_gaussian_log(args.input_path, args.output_path, args.verbose)

        if Path(args.input_path).is_file():  # 单个文件处理结果
            print("\n处理完成！")
            for k, v in result.items():
                print(k, v)
        else:  # 目录处理结果
            print(
                f"\n处理完成！共处理 {result['total_files']} 个文件，成功 {result['processed_files']} 个，失败 {result['failed_files']} 个"
            )
            # 如果有失败的文件，显示详细信息
            if result["failed_files"] > 0:
                print(f"\n失败文件详情:")
                for i, failed_file in enumerate(result["failed_file_list"], 1):
                    print(f"  {i}. {failed_file['file_path']}")
                    print(f"     错误: {failed_file['error_message']}")

    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
