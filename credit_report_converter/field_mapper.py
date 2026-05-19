"""字段映射模块：读取字段字典和码值表，生成映射规则"""

import pandas as pd
from typing import Dict, Any


class FieldMapper:
    """字段映射器，负责加载字段字典和码值表，提供字段/码值翻译能力"""

    def __init__(self, field_dict_path: str, code_value_path: str):
        """
        初始化字段映射器

        Args:
            field_dict_path: 字段字典xlsx文件路径
            code_value_path: 码值表xlsx文件路径
        """
        self.field_mapping: Dict[str, str] = {}
        self.code_value_mapping: Dict[str, Dict[str, str]] = {}
        self.field_to_keyword: Dict[str, str] = {}
        self._load_field_dict(field_dict_path)
        self._load_code_values(code_value_path)

    def _load_field_dict(self, path: str) -> None:
        """加载字段字典，生成字段编码到英文名的映射，以及字段到码值关键字的映射

        同时存储原始大小写和大写版本的键，以支持大小写不敏感查找。
        """
        df = pd.read_excel(path)

        # 只保留有英文名的字段
        df = df.dropna(subset=["英文名"])

        for _, row in df.iterrows():
            field_code = row["字段"]
            english_name = row["英文名"]
            if field_code and english_name:
                self.field_mapping[field_code] = english_name
                # 同时存储大写版本，支持大小写不敏感查找
                upper_code = field_code.upper()
                if upper_code != field_code:
                    self.field_mapping[upper_code] = english_name

            # 记录字段到码值关键字的映射
            keyword = row.get("对应个人征信码值表关键字")
            if field_code and pd.notna(keyword):
                self.field_to_keyword[field_code] = keyword
                upper_code = field_code.upper()
                if upper_code != field_code:
                    self.field_to_keyword[upper_code] = keyword

    def _load_code_values(self, path: str) -> None:
        """加载码值表，生成关键字 -> {代码 -> 英文名称} 的映射"""
        df = pd.read_excel(path)

        for _, row in df.iterrows():
            keyword = row["关键字"]
            code = str(row["代码"])
            english_name = row["英文名称"]

            if keyword not in self.code_value_mapping:
                self.code_value_mapping[keyword] = {}

            if pd.notna(code) and pd.notna(english_name):
                self.code_value_mapping[keyword][code] = english_name

    def get_field_name(self, field_code: str) -> str:
        """获取字段编码对应的英文名（大小写不敏感）"""
        # 先尝试原始大小写，再尝试大写
        result = self.field_mapping.get(field_code)
        if result is not None:
            return result
        return self.field_mapping.get(field_code.upper(), field_code)

    def get_code_value(self, code_key: str, code_value: str) -> str:
        """获取码值对应的英文名"""
        if code_key in self.code_value_mapping:
            return self.code_value_mapping[code_key].get(code_value, code_value)
        return code_value

    def translate_value(self, field_code: str, value: Any) -> str:
        """翻译字段值（如果是码值则翻译为英文），字段编码大小写不敏感"""
        if not value or value == "":
            return ""

        # 通过字段到关键字的映射查找对应的码值表，先尝试原始大小写，再尝试大写
        keyword = self.field_to_keyword.get(field_code)
        if keyword is None:
            keyword = self.field_to_keyword.get(field_code.upper())
        if keyword:
            return self.get_code_value(keyword, str(value))

        return str(value)
