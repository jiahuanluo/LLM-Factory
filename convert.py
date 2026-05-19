#!/usr/bin/env python
"""征信报告JSON转文本转换工具主入口"""

import argparse
import os
from credit_report_converter import BatchProcessor


def main():
    parser = argparse.ArgumentParser(description="征信报告JSON转文本转换工具")
    parser.add_argument(
        "--input", "-i",
        help="输入目录（包含JSON文件）"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="输出目录（存放转换后的txt文件）"
    )
    parser.add_argument(
        "--dict-file",
        default="moc_data/credit_report_dict.json",
        help="合并后的JSON字典文件路径（默认）"
    )
    parser.add_argument(
        "--field-dict",
        help="字段字典xlsx文件路径（兼容旧版）"
    )
    parser.add_argument(
        "--code-value",
        help="码值表xlsx文件路径（兼容旧版）"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="并行工作进程数"
    )
    parser.add_argument(
        "--single-file",
        help="处理单个文件（指定JSON文件路径）"
    )

    args = parser.parse_args()

    # 验证参数
    if not args.single_file and not args.input:
        parser.error("必须指定 --input 或 --single-file")

    # 创建处理器
    if args.field_dict and args.code_value:
        processor = BatchProcessor(
            field_dict_path=args.field_dict,
            code_value_path=args.code_value,
            num_workers=args.workers,
        )
    else:
        processor = BatchProcessor(
            json_dict_path=args.dict_file,
            num_workers=args.workers,
        )

    if args.single_file:
        # 处理单个文件
        output_file = args.output
        if os.path.isdir(output_file):
            filename = os.path.basename(args.single_file).replace(".json", ".txt")
            output_file = os.path.join(output_file, filename)

        processor.process_file(args.single_file, output_file)
        print(f"Processed {args.single_file} -> {output_file}")
    else:
        # 处理目录
        processor.process_directory(args.input, args.output)
        print(f"Batch processing completed: {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
