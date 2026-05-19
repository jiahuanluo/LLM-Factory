import json
import pytest
from credit_report_converter.json_parser import JsonParser


def test_json_parser_load():
    """测试JsonParser能否加载JSON文件"""
    parser = JsonParser(excluded_fields={"name", "certNo"})
    with open("moc_data/CrisPbc.json", "r") as f:
        data = json.load(f)
    result = parser.parse(data)
    assert result is not None
    assert "header" in result


def test_excluded_fields():
    """测试排除字段是否被移除"""
    parser = JsonParser(excluded_fields={"name", "certNo"})
    data = {
        "name": "test",
        "certNo": "123456",
        "gender": "1",
        "birthday": "1981-08-15"
    }
    result = parser.parse(data)
    assert "name" not in result
    assert "certNo" not in result
    assert "gender" in result
    assert "birthday" in result


def test_nested_fields():
    """测试嵌套字段的解析"""
    parser = JsonParser(excluded_fields={"name"})
    data = {
        "personInfo": {
            "identity": {
                "gender": "1",
                "birthday": "1981-08-15",
                "name": "test"
            }
        }
    }
    result = parser.parse(data)
    assert "personInfo" in result
    assert "identity" in result["personInfo"]
    assert "gender" in result["personInfo"]["identity"]
    assert "name" not in result["personInfo"]["identity"]
