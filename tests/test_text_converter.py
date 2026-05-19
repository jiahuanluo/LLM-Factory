import pytest
from credit_report_converter.text_converter import TextConverter
from credit_report_converter.field_mapper import FieldMapper


@pytest.fixture
def converter():
    """创建转换器fixture"""
    mapper = FieldMapper(
        field_dict_path="moc_data/个人征信DB表结构字典.xlsx",
        code_value_path="moc_data/个人征信码值表.xlsx"
    )
    return TextConverter(field_mapper=mapper)


def test_simple_conversion(converter):
    """测试简单字段转换（使用实际字段编码）"""
    data = {
        "tranDate": "20250124",
        "pb01ad01": "1",        # gender
        "pb01ar01": "1981-08-15"  # date_of_birth
    }
    result = converter.convert(data)
    assert "tranDate=20250124" in result
    assert "gender=male" in result
    assert "date_of_birth=1981-08-15" in result


def test_special_tokens(converter):
    """测试special tokens"""
    data = {
        "header": {"tranDate": "20250124"},
        "personInfo": {"pb01ad01": "1"}
    }
    result = converter.convert(data)
    assert "[CLS]" in result
    assert "[HDR]" in result
    assert "[PERS]" in result
    assert "[SEP]" in result


def test_account_conversion(converter):
    """测试账户信息转换"""
    data = {
        "accountInfos": [
            {
                "accountBasic": {
                    "pd01ad01": "D1",     # account_type -> non-revolving_loan_account
                    "pd01ad02": "11",      # business_management_institution_type -> commercial_bank
                    "pd01aj01": "360000"   # loan_amount (no code value mapping)
                }
            }
        ]
    }
    result = converter.convert(data)
    assert "[ACCT]" in result
    assert "account_type=non-revolving_loan_account" in result


def test_empty_data(converter):
    """测试空数据"""
    data = {}
    result = converter.convert(data)
    assert result == "[CLS]"


def test_section_sep_tokens(converter):
    """测试每个section后都有[SEP]"""
    data = {
        "header": {"reportsn": "mock_202501212122362152"},
        "personInfo": {"pb01ad01": "1"},
    }
    result = converter.convert(data)
    # 每个section token后面的数据，最后都跟[SEP]
    assert result.count("[SEP]") >= 2


def test_multiple_accounts(converter):
    """测试多个账户"""
    data = {
        "accountInfos": [
            {"accountBasic": {"pd01ad01": "D1"}},
            {"accountBasic": {"pd01ad01": "R2"}},
        ]
    }
    result = converter.convert(data)
    assert result.count("[ACCT]") == 2


def test_unknown_fields_pass_through(converter):
    """测试未知字段原样传递"""
    data = {"unknownField": "someValue"}
    result = converter.convert(data)
    assert "unknownField=someValue" in result


def test_nested_dict_flattening(converter):
    """测试嵌套字典展平"""
    data = {
        "header": {
            "request": {
                "reportsn": "mock_123"
            }
        }
    }
    result = converter.convert(data)
    assert "request_report_number=mock_123" in result
