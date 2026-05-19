import pytest
from credit_report_converter.field_mapper import FieldMapper


JSON_DICT_PATH = "moc_data/credit_report_dict.json"


@pytest.fixture(scope="module")
def mapper():
    """创建FieldMapper fixture（使用JSON字典）"""
    return FieldMapper(json_path=JSON_DICT_PATH)


def test_field_mapper_load_json():
    """测试FieldMapper能否从JSON加载"""
    mapper = FieldMapper(json_path=JSON_DICT_PATH)
    assert mapper is not None
    assert len(mapper.field_mapping) > 0
    assert len(mapper.code_value_mapping) > 0


def test_field_mapper_load_xlsx():
    """测试FieldMapper能否从xlsx加载（兼容旧版）"""
    mapper = FieldMapper(
        field_dict_path="moc_data/个人征信DB表结构字典.xlsx",
        code_value_path="moc_data/个人征信码值表.xlsx",
    )
    assert mapper is not None
    assert len(mapper.field_mapping) > 0


def test_field_mapping(mapper):
    """测试字段编码到英文名的映射"""
    assert "PA01DQ01" in mapper.field_mapping
    assert mapper.field_mapping["PA01DQ01"] == "anti_fraud_warning_flag"


def test_code_value_mapping(mapper):
    """测试码值到英文名的映射"""
    assert "个人证件类型代码表" in mapper.code_value_mapping
    assert mapper.code_value_mapping["个人证件类型代码表"]["1"] == "household_book"


def test_get_field_name(mapper):
    """测试get_field_name方法"""
    assert mapper.get_field_name("PA01DQ01") == "anti_fraud_warning_flag"
    assert mapper.get_field_name("unknown_field") == "unknown_field"


def test_get_code_value(mapper):
    """测试get_code_value方法"""
    assert mapper.get_code_value("个人证件类型代码表", "1") == "household_book"
    assert mapper.get_code_value("个人证件类型代码表", "ZZ") == "ZZ"
    assert mapper.get_code_value("不存在的", "1") == "1"


def test_translate_value(mapper):
    """测试translate_value方法"""
    assert mapper.translate_value("PA01DQ01", "") == ""
    assert mapper.translate_value("PA01DQ01", None) == ""
    assert mapper.translate_value("reportsn", "ABC123") == "ABC123"
    assert mapper.translate_value("PA01CD01", "1") == "household_book"


def test_field_to_keyword_mapping(mapper):
    """测试字段编码到码值关键字的映射"""
    assert "PA01CD01" in mapper.field_to_keyword
    assert mapper.field_to_keyword["PA01CD01"] == "个人证件类型代码表"
