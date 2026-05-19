import os
import json
import tempfile
import pytest
from credit_report_converter.batch_processor import BatchProcessor


JSON_DICT_PATH = "moc_data/credit_report_dict.json"


def test_single_file_processing():
    """测试单文件处理"""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(input_dir)
        os.makedirs(output_dir)

        test_data = {"header": {"tranDate": "20250124"}, "personInfo": {"gender": "1"}}
        with open(os.path.join(input_dir, "test.json"), "w") as f:
            json.dump(test_data, f)

        processor = BatchProcessor(json_dict_path=JSON_DICT_PATH)
        processor.process_directory(input_dir, output_dir)

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

        for i in range(5):
            test_data = {"header": {"tranDate": f"2025012{i}"}}
            with open(os.path.join(input_dir, f"test_{i}.json"), "w") as f:
                json.dump(test_data, f)

        processor = BatchProcessor(json_dict_path=JSON_DICT_PATH, num_workers=2)
        processor.process_directory(input_dir, output_dir)

        output_files = os.listdir(output_dir)
        assert len(output_files) == 5
