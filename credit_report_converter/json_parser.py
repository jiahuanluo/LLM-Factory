"""JSON解析模块：解析征信报告JSON，过滤敏感字段"""

import json
from typing import Dict, Any, Set

from .config import EXCLUDED_FIELDS


class JsonParser:
    """JSON解析器"""

    def __init__(self, excluded_fields: Set[str] = None):
        """
        初始化JSON解析器

        Args:
            excluded_fields: 需要排除的字段集合，如果为None则使用默认配置
        """
        self.excluded_fields = excluded_fields or EXCLUDED_FIELDS

    def parse(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析JSON数据，过滤敏感字段

        Args:
            data: 原始JSON数据

        Returns:
            过滤后的数据
        """
        return self._filter_dict(data)

    def _filter_dict(self, d: Any) -> Any:
        """递归过滤字典中的敏感字段"""
        if not isinstance(d, dict):
            return d

        result = {}
        for key, value in d.items():
            # 检查字段是否需要排除
            if key in self.excluded_fields:
                continue

            # 递归处理嵌套结构
            if isinstance(value, dict):
                result[key] = self._filter_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    self._filter_dict(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[key] = value

        return result

    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        解析JSON文件

        Args:
            file_path: JSON文件路径

        Returns:
            解析后的数据
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return self.parse(data)
