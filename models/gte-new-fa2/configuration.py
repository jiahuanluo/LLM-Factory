# coding=utf-8
# Copyright 2024 The GTE Team Authors and Alibaba Group.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""NEW model configuration (FA2/SDPA clean version, no unpad/pack_qkv)."""
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class NewConfig(PretrainedConfig):
    model_type = "new"

    def __init__(
        self,
        vocab_size=30528,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=2048,
        type_vocab_size=1,
        initializer_range=0.02,
        layer_norm_type='layer_norm',
        layer_norm_eps=1e-12,
        position_embedding_type="rope",
        rope_theta=10000.0,
        rope_scaling=None,
        classifier_dropout=None,
        logn_attention_scale=False,
        logn_attention_clip1=False,
        **kwargs,
    ):
        # HF 5.x 在 super().__init__() 里就跑 RoPE validator（读 max_position_embeddings 等），
        # 必须先把自己关心的字段赋值，再调 super。
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_type = layer_norm_type
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.classifier_dropout = classifier_dropout
        self.logn_attention_scale = logn_attention_scale
        self.logn_attention_clip1 = logn_attention_clip1
        super().__init__(**kwargs)
