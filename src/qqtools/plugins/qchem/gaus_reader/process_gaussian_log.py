#!/usr/bin/env python3
"""
高斯Log文件处理器
从高斯log文件中提取关键信息并保存为JSON格式
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


def process_gaussian_log(input_file, output_file=None):
    """
    处理高斯log文件并提取信息

    Args:
        input_file (str): 输入的log文件路径
        output_file (str, optional): 输出的JSON文件路径

    Returns:
        dict: 提取的结果数据
    """

    # 验证输入文件
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    if not input_path.suffix.lower() in [".log", ".out"]:
        print(f"警告：输入文件扩展名不是.log或.out，但仍将尝试处理: {input_file}")

    print(f"正在处理文件: {input_file}")

    # 创建读取器并处理文件
    try:
        reader = create_g16_reader(False)
        results = reader.read_file(str(input_path))

        results = convert_numpy_types(results)
        print(f"成功提取 {len(results)} 个数据项")

        # for k, v in results.items():
        #     print(k, " ", v)

    except Exception as e:
        print(f"处理文件时出错: {e}")
        traceback.print_exc()

    # 确定输出文件路径
    if output_file is None:
        # 自动生成输出文件名（与输入文件同名，扩展名为.json）
        output_path = input_path.parent / (input_path.stem + ".json")
        print(f"未指定输出文件，将保存到: {output_path}")
    else:
        output_path = Path(output_file)
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"结果已保存到: {output_path}")
        print(f"文件大小: {output_path.stat().st_size} bytes")

    except Exception as e:
        print(f"保存结果时出错")
        raise e

    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="处理高斯log文件并提取关键信息为JSON格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s calculation.log
  %(prog)s calculation.log results.json
  %(prog)s /path/to/calculation.log /output/path/results.json
        """,
    )

    parser.add_argument("input_file", help="输入的高斯log文件路径")
    parser.add_argument(
        "output_file", nargs="?", default=None, help="输出的JSON文件路径（可选，默认生成同名.json文件）"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出模式")

    args = parser.parse_args()

    try:
        # 处理文件
        results = process_gaussian_log(args.input_file, args.output_file)
        print("\n处理完成！")

    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
