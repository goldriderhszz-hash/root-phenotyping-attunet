import numpy as np

from root_phenotyping import AttentionUNet, extract_phenotypes_decoupled


def test_model_parameter_count():
    model = AttentionUNet()
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 7_981_277


def test_empty_mask_contract():
    mask = np.zeros((32, 32), dtype=np.uint8)
    assert extract_phenotypes_decoupled(mask) == (0, 0, 0, 0)
