"""批量处理模块：支持多进程并行处理多个JSON文件"""

import os
import json
from typing import List, Tuple
from multiprocessing import Pool
from .field_mapper import FieldMapper
from .json_parser import JsonParser
from .text_converter import TextConverter
from .config import EXCLUDED_FIELDS


class BatchProcessor:
    """批量处理器"""

    def __init__(
        self,
        field_dict_path: str,
        code_value_path: str,
        excluded_fields: set = None,
        num_workers: int = 4
    ):
        """
        初始化批量处理器

        Args:
            field_dict_path: 字段字典路径
            code_value_path: 码值表路径
            excluded_fields: 排除字段集合
            num_workers: 并行工作进程数
        """
        self.field_mapper = FieldMapper(field_dict_path, code_value_path)
        self.json_parser = JsonParser(excluded_fields or EXCLUDED_FIELDS)
        self.text_converter = TextConverter(self.field_mapper)
        self.num_workers = num_workers

    def _process_single_file(self, args: Tuple[str, str]) -> bool:
        """处理单个文件"""
        input_path, output_path = args
        try:
            # 读取JSON
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 解析并过滤
            parsed_data = self.json_parser.parse(data)

            # 转换为文本
            text = self.text_converter.convert(parsed_data)

            # 写入输出文件
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)

            return True
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            return False

    def process_directory(self, input_dir: str, output_dir: str):
        """
        处理目录中的所有JSON文件

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 收集所有JSON文件
        json_files = []
        for filename in os.listdir(input_dir):
            if filename.endswith(".json"):
                input_path = os.path.join(input_dir, filename)
                output_filename = filename.replace(".json", ".txt")
                output_path = os.path.join(output_dir, output_filename)
                json_files.append((input_path, output_path))

        if not json_files:
            print(f"No JSON files found in {input_dir}")
            return

        print(f"Processing {len(json_files)} files with {self.num_workers} workers...")

        # 多进程处理
        with Pool(self.num_workers) as pool:
            results = pool.map(self._process_single_file, json_files)

        # 统计结果
        success_count = sum(results)
        print(f"Successfully processed {success_count}/{len(json_files)} files")

    def process_file(self, input_path: str, output_path: str):
        """处理单个文件"""
        self._process_single_file((input_path, output_path))
