import pytest
from credit_report_converter.field_mapper import FieldMapper


def test_field_mapper_load():
    """测试FieldMapper能否加载字段字典和码值表"""
    mapper = FieldMapper(
        field_dict_path="moc_data/个人征信DB表结构字典.xlsx",
        code_value_path="moc_data/个人征信码值表.xlsx"
    )
    assert mapper is not None
    assert len(mapper.field_mapping) > 0
    assert len(mapper.code_value_mapping) > 0


def test_field_mapping():
    """测试字段编码到英文名的映射"""
    mapper = FieldMapper(
        field_dict_path="moc_data/个人征信DB表结构字典.xlsx",
        code_value_path="moc_data/个人征信码值表.xlsx"
    )
    # 测试已知的映射
    assert "PA01DQ01" in mapper.field_mapping
    assert mapper.field_mapping["PA01DQ01"] == "anti_fraud_warning_flag"


def test_code_value_mapping():
    """测试码值到英文名的映射"""
    mapper = FieldMapper(
        field_dict_path="moc_data/个人征信DB表结构字典.xlsx",
        code_value_path="moc_data/个人征信码值表.xlsx"
    )
    # 测试证件类型码值
    assert "个人证件类型代码表" in mapper.code_value_mapping
    assert mapper.code_value_mapping["个人证件类型代码表"]["1"] == "household_book"


def test_get_field_name():
    """测试get_field_name方法"""
    mapper = FieldMapper(
        field_dict_path="moc_data/个人征信DB表结构字典.xlsx",
        code_value_path="moc_data/个人征信码值表.xlsx"
    )
    # 已知字段
    assert mapper.get_field_name("PA01DQ01") == "anti_fraud_warning_flag"
    # 未知字段返回原值
    assert mapper.get_field_name("unknown_field") == "unknown_field"


def test_get_code_value():
    """测试get_code_value方法"""
    mapper = FieldMapper(
        field_dict_path="moc_data/个人征信DB表结构字典.xlsx",
        code_value_path="moc_data/个人征信码值表.xlsx"
    )
    # 已知码值
    assert mapper.get_code_value("个人证件类型代码表", "1") == "household_book"
    # 未知码值返回原值
    assert mapper.get_code_value("个人证件类型代码表", "ZZ") == "ZZ"
    # 未知关键字返回原值
    assert mapper.get_code_value("不存在的", "1") == "1"


def test_translate_value():
    """测试translate_value方法"""
    mapper = FieldMapper(
        field_dict_path="moc_data/个人征信DB表结构字典.xlsx",
        code_value_path="moc_data/个人征信码值表.xlsx"
    )
    # 空值返回空字符串
    assert mapper.translate_value("PA01DQ01", "") == ""
    assert mapper.translate_value("PA01DQ01", None) == ""

    # 非码值字段返回原值
    result = mapper.translate_value("reportsn", "ABC123")
    assert result == "ABC123"

    # 码值字段翻译
    result = mapper.translate_value("PA01CD01", "1")
    assert result == "household_book"


def test_field_to_keyword_mapping():
    """测试字段编码到码值关键字的映射"""
    mapper = FieldMapper(
        field_dict_path="moc_data/个人征信DB表结构字典.xlsx",
        code_value_path="moc_data/个人征信码值表.xlsx"
    )
    # PA01CD01 对应 个人证件类型代码表
    assert "PA01CD01" in mapper.field_to_keyword
    assert mapper.field_to_keyword["PA01CD01"] == "个人证件类型代码表"
