"""测试 modeling.py 重构后的行为一致性"""

import importlib.util
import os
import sys

import pytest
import torch

# 目录名含连字符，无法直接用 Python import，需通过 importlib 加载
_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
_MODEL_DIR = os.path.join(_PROJECT_ROOT, "models", "gte-large-en-v1.5")


def _load_module(name: str):
    """从 gte-large-en-v1.5 目录按文件名加载模块。"""
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_MODEL_DIR, f"{name}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    # 让模块内的相对导入（from .configuration import ...）正常工作
    mod.__package__ = "models.gte_large_en_v1_5"
    sys.modules[f"models.gte_large_en_v1_5.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


# 先加载 configuration（modeling 依赖它）
_configuration = _load_module("configuration")
_modeling = _load_module("modeling")

NewConfig = _configuration.NewConfig
NewModel = _modeling.NewModel
NewEmbeddings = _modeling.NewEmbeddings
unpad_input = _modeling.unpad_input
pad_input = _modeling.pad_input
IndexFirstAxis = _modeling.IndexFirstAxis
IndexPutFirstAxis = _modeling.IndexPutFirstAxis


@pytest.fixture
def config():
    """创建测试用配置"""
    return NewConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=128,
        unpad_inputs=False,
        use_memory_efficient_attention=False,
    )


@pytest.fixture
def batch_input():
    """创建测试用批次输入"""
    batch_size, seq_length = 2, 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
    # 第二个序列只有 8 个有效 token
    attention_mask[1, 8:] = 0
    return input_ids, attention_mask


class TestUnpadInput:
    """测试 unpad_input 函数"""

    def test_unpad_with_attention_mask(self):
        """测试使用 attention_mask 进行 unpad"""
        hidden_states = torch.randn(2, 10, 768)
        attention_mask = torch.tensor([
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        ])

        result = unpad_input(hidden_states, attention_mask)

        assert result.shape == (8, 768)

    def test_unpad_with_indices(self):
        """测试使用预计算索引进行 unpad"""
        hidden_states = torch.randn(2, 10, 768)
        indices = torch.tensor([0, 1, 2, 10, 11, 12, 13, 14])

        result = unpad_input(hidden_states, indices=indices)

        assert result.shape == (8, 768)

    def test_unpad_raises_without_args(self):
        """测试没有参数时抛出异常"""
        hidden_states = torch.randn(2, 10, 768)

        with pytest.raises(AssertionError):
            unpad_input(hidden_states)


class TestPadInput:
    """测试 pad_input 函数"""

    def test_pad_input(self):
        """测试 pad_input 函数"""
        inputs = torch.randn(8, 768)
        indices = torch.tensor([0, 1, 2, 10, 11, 12, 13, 14])

        result = pad_input(inputs, indices, 2, 10)

        assert result.shape == (2, 10, 768)
        # 验证 padding 位置为 0
        assert result[0, 3:].sum() == 0
        assert result[1, 5:].sum() == 0


class TestNewModelForward:
    """测试 NewModel.forward() 方法"""

    def test_forward_without_unpad(self, config, batch_input):
        """测试不使用 unpad 的前向传播"""
        input_ids, attention_mask = batch_input
        model = NewModel(config)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            unpad_inputs=False,
        )

        assert outputs.last_hidden_state.shape == (2, 16, 256)

    def test_forward_with_unpad(self, config, batch_input):
        """测试使用 unpad 的前向传播"""
        input_ids, attention_mask = batch_input
        config.unpad_inputs = True
        model = NewModel(config)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            unpad_inputs=True,
        )

        # unpad 后仍应返回 padded 输出（因为 length=None）
        assert outputs.last_hidden_state.shape == (2, 16, 256)
