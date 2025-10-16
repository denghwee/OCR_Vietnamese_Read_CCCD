from dataclasses import dataclass

@dataclass
class OCRConfig:
    lang: str = "vi"
    model_dir: str = "models/"

@dataclass
class TemplateConfig:
    coords = {
        "id": (100, 50, 300, 60),
        "name": (100, 120, 500, 60),
        "dob": (100, 200, 200, 60),
        "address": (100, 200, 600, 100),
    }

CONFIG = {
    "detector": {
        "type": "paddleocr",
        "lang": "vi",
        "device": "gpu"
    },
    "recognizer": {
        "type": "vietocr",
        "config_name": "vgg_transformer",
        "device": "cuda:0"
    },
    "aligner": {
        "type": "default"
    },
    "cleaner": {
        "type": "default"
    },
    "api_key": {
        "id": "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"
    }
}