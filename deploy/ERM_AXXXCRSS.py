"""ERM_AXXXCRSS — 二分类风控模型生产部署入口（多模型版本）。

典型用法:
    from deploy.ERM_AXXXCRSS import ERM_AXXXCRSS

    model = ERM_AXXXCRSS()
    model.init_metadate()
    result = model.predict({"data": [{"text": "样本文本"}, {"text": "另一段"}]})
    # result == {"probabilities": {"model_a": [...], "model_b": [...]}}
"""

import threading

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ERM_AXXXCRSS:
    MODEL_PATHS = {
        "model_a": "/path/to/your/trained/model_a",
        "model_b": "/path/to/your/trained/model_b",
    }
    MAX_SEQ_LENGTH = 128

    def __init__(self) -> None:
        self.models = {}
        self.tokenizers = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()
        self._initialized = False

    def init_metadate(self) -> None:
        for name, path in self.MODEL_PATHS.items():
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSequenceClassification.from_pretrained(path)
            model.to(self.device)
            model.eval()
            self.tokenizers[name] = tokenizer
            self.models[name] = model
        self._initialized = True

    def predict(self, request: dict) -> dict:
        if not self._initialized:
            raise RuntimeError("模型未初始化，请先调用 init_metadate()")

        data = request.get("data", [])
        if not data:
            return {"probabilities": {name: [] for name in self.models}}

        texts = [item["text"] for item in data]

        with self._lock:
            results = {}
            for name, model in self.models.items():
                tokenizer = self.tokenizers[name]
                with torch.no_grad():
                    inputs = tokenizer(
                        texts,
                        padding=True,
                        truncation=True,
                        max_length=self.MAX_SEQ_LENGTH,
                        return_tensors="pt",
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    logits = model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1)
                    results[name] = probs[:, 1].cpu().tolist()

        return {"probabilities": results}
