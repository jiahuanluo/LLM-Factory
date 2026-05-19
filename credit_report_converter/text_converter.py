"""文本转换模块：将解析后的数据转换为英文键值对文本"""

from typing import Dict, Any, List

from .field_mapper import FieldMapper
from .config import SPECIAL_TOKENS


class TextConverter:
    """文本转换器，将解析后的征信报告数据转换为带special tokens的键值对文本"""

    def __init__(self, field_mapper: FieldMapper):
        """
        初始化文本转换器

        Args:
            field_mapper: 字段映射器实例
        """
        self.field_mapper = field_mapper

    def convert(self, data: Dict[str, Any]) -> str:
        """
        将数据转换为文本串

        Args:
            data: 解析后的数据

        Returns:
            转换后的文本串
        """
        parts = []
        parts.append(SPECIAL_TOKENS["CLS"])

        # 处理报告头
        if "header" in data:
            header_text = self._convert_section(data["header"])
            parts.append(f"{SPECIAL_TOKENS['HDR']} {header_text}")
            parts.append(SPECIAL_TOKENS["SEP"])

        # 处理个人信息
        if "personInfo" in data:
            pers_text = self._convert_section(data["personInfo"])
            parts.append(f"{SPECIAL_TOKENS['PERS']} {pers_text}")
            parts.append(SPECIAL_TOKENS["SEP"])

        # 处理汇总信息
        if "summaryInfo" in data:
            summ_text = self._convert_section(data["summaryInfo"])
            parts.append(f"{SPECIAL_TOKENS['SUMM']} {summ_text}")
            parts.append(SPECIAL_TOKENS["SEP"])

        # 处理账户信息（每个账户一个）
        if "accountInfos" in data:
            for account in data["accountInfos"]:
                acct_text = self._convert_section(account)
                parts.append(f"{SPECIAL_TOKENS['ACCT']} {acct_text}")
                parts.append(SPECIAL_TOKENS["SEP"])

        # 处理公共信息
        if "publicInfo" in data:
            pub_text = self._convert_section(data["publicInfo"])
            parts.append(f"{SPECIAL_TOKENS['PUB']} {pub_text}")
            parts.append(SPECIAL_TOKENS["SEP"])

        # 处理查询记录
        if "queryRecords" in data:
            for query in data["queryRecords"]:
                query_text = self._convert_section(query)
                parts.append(f"{SPECIAL_TOKENS['QUERY']} {query_text}")
                parts.append(SPECIAL_TOKENS["SEP"])

        # 顶层字段（非section的字段）
        section_keys = {
            "header", "personInfo", "summaryInfo",
            "accountInfos", "publicInfo", "queryRecords"
        }
        top_level = {k: v for k, v in data.items() if k not in section_keys}
        if top_level:
            kv_pairs = []
            self._flatten_dict(top_level, "", kv_pairs)
            parts.extend(kv_pairs)

        return " ".join(parts)

    def _convert_section(self, data: Dict[str, Any]) -> str:
        """转换一个section的数据为键值对字符串"""
        kv_pairs = []
        self._flatten_dict(data, "", kv_pairs)
        return " ".join(kv_pairs)

    def _flatten_dict(self, d: Dict[str, Any], prefix: str, kv_pairs: List[str]):
        """递归展平字典为键值对列表

        对于嵌套结构，output key使用 prefix + translated_name 格式，
        翻译时使用原始字段编码（不含前缀）以匹配字段字典。
        """
        for key, value in d.items():
            if isinstance(value, dict):
                self._flatten_dict(value, f"{prefix}{key}_", kv_pairs)
            elif isinstance(value, list):
                for i, item in enumerate(value, 1):
                    if isinstance(item, dict):
                        self._flatten_dict(item, f"{prefix}{key}_{i}_", kv_pairs)
                    else:
                        # 列表项：prefix已包含 key_i_ 前缀，直接用翻译后的字段名
                        translated_name = self.field_mapper.get_field_name(key)
                        translated_value = self.field_mapper.translate_value(
                            key, item
                        )
                        if translated_value:
                            kv_pairs.append(
                                f"{prefix}{translated_name}={translated_value}"
                            )
            else:
                # 用原始字段编码翻译，输出时用 prefix + translated_name
                translated_name = self.field_mapper.get_field_name(key)
                translated_value = self.field_mapper.translate_value(key, value)
                if translated_value:
                    kv_pairs.append(
                        f"{prefix}{translated_name}={translated_value}"
                    )
