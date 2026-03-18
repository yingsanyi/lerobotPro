import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.s2_pi05 import S2PI05Config, S2PI05Policy
from lerobot.utils.constants import (
    OBS_SEMANTIC_BOXES_XYXY,
    OBS_SEMANTIC_OCR_CONF,
    OBS_SEMANTIC_OCR_TEXT,
    OBS_SEMANTIC_TARGET_INDEX,
)


def main():
    config = S2PI05Config(device="cpu", semantic_require_inputs=True)
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }

    dataset_stats = {
        "observation.state": {
            "mean": torch.zeros(14),
            "std": torch.ones(14),
            "min": torch.zeros(14),
            "max": torch.ones(14),
            "q01": torch.zeros(14),
            "q99": torch.ones(14),
        },
        "action": {
            "mean": torch.zeros(7),
            "std": torch.ones(7),
            "min": torch.zeros(7),
            "max": torch.ones(7),
            "q01": torch.zeros(7),
            "q99": torch.ones(7),
        },
        "observation.images.base_0_rgb": {
            "mean": torch.zeros(3, 224, 224),
            "std": torch.ones(3, 224, 224),
            "q01": torch.zeros(3, 224, 224),
            "q99": torch.ones(3, 224, 224),
        },
    }

    policy = S2PI05Policy(config)
    preprocessor, _ = make_pre_post_processors(config, dataset_stats=dataset_stats)

    batch = {
        "observation.state": torch.randn(1, 14),
        "observation.images.base_0_rgb": torch.rand(1, 3, 224, 224),
        "action": torch.randn(1, config.chunk_size, 7),
        OBS_SEMANTIC_BOXES_XYXY: torch.tensor(
            [[[0.05, 0.10, 0.30, 0.40], [0.42, 0.22, 0.70, 0.64], [0.72, 0.18, 0.92, 0.52]]],
            dtype=torch.float32,
        ),
        OBS_SEMANTIC_OCR_TEXT: [["pain relief", "cold medicine", "cup"]],
        OBS_SEMANTIC_OCR_CONF: torch.tensor([[0.95, 0.78, 0.25]], dtype=torch.float32),
        OBS_SEMANTIC_TARGET_INDEX: torch.tensor([0], dtype=torch.long),
        "task": ["Pick up the medicine with the matching label"],
    }
    batch = preprocessor(batch)
    loss, loss_dict = policy.forward(batch)
    print(loss.item(), loss_dict)


if __name__ == "__main__":
    main()
