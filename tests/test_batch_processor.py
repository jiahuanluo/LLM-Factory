import os
import json
import tempfile
import pytest
from credit_report_converter.batch_processor import BatchProcessor


def test_single_file_processing():
    """测试单文件处理"""
    # 创建临时输入目录
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(input_dir)
        os.makedirs(output_dir)

        # 创建测试JSON文件
        test_data = {"header": {"tranDate": "20250124"}, "personInfo": {"gender": "1"}}
        with open(os.path.join(input_dir, "test.json"), "w") as f:
            json.dump(test_data, f)

        # 处理
        processor = BatchProcessor(
            field_dict_path="moc_data/个人征信DB表结构字典.xlsx",
            code_value_path="moc_data/个人征信码值表.xlsx"
        )
        processor.process_directory(input_dir, output_dir)

        # 验证输出
        assert os.path.exists(os.path.join(output_dir, "test.txt"))
        with open(os.path.join(output_dir, "test.txt"), "r") as f:
            content = f.read()
        assert "[CLS]" in content
        assert "[HDR]" in content


def test_multiprocessing():
    """测试多进程处理"""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(input_dir)
        os.makedirs(output_dir)

        # 创建多个测试文件
        for i in range(5):
            test_data = {"header": {"tranDate": f"2025012{i}"}}
            with open(os.path.join(input_dir, f"test_{i}.json"), "w") as f:
                json.dump(test_data, f)

        # 多进程处理
        processor = BatchProcessor(
            field_dict_path="moc_data/个人征信DB表结构字典.xlsx",
            code_value_path="moc_data/个人征信码值表.xlsx",
            num_workers=2
        )
        processor.process_directory(input_dir, output_dir)

        # 验证所有文件都被处理
        output_files = os.listdir(output_dir)
        assert len(output_files) == 5
