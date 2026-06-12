import numpy as np
import pytest
import torch

from heat.model import HEAT
from heat.models.corner_to_edge import get_all_combinations, get_infer_edge_pairs


def test_force_cpu_model_is_not_wrapped_in_data_parallel():
    model = HEAT(force_cpu=True)

    assert model.device.type == "cpu"
    assert not isinstance(model._backbone, torch.nn.DataParallel)


def test_infer_batch_requires_checkpoint_before_inference():
    model = HEAT(force_cpu=True)
    image = np.zeros((256, 256, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="load_checkpoint"):
        model.infer_batch([image])


def test_infer_batch_rejects_empty_image_list_after_checkpoint_loaded():
    model = HEAT(force_cpu=True)
    model._checkpoint_args = object()

    with pytest.raises(ValueError, match="at least one image"):
        model.infer_batch([])


def test_infer_batch_validates_image_shape_and_dtype():
    model = HEAT(force_cpu=True)
    model._checkpoint_args = object()

    with pytest.raises(TypeError, match="dtype uint8"):
        model.infer_batch([np.zeros((256, 256, 3), dtype=np.float32)])

    with pytest.raises(ValueError, match=r"shape \(256, 256, 3\)"):
        model.infer_batch([np.zeros((128, 128, 3), dtype=np.uint8)])


def test_edge_combinations_are_generated_beyond_legacy_limit():
    combinations = get_all_combinations(351)

    assert combinations.shape == (61425, 2)
    np.testing.assert_array_equal(combinations[0], np.array([0, 1]))
    np.testing.assert_array_equal(combinations[-1], np.array([349, 350]))


def test_get_infer_edge_pairs_uses_dynamic_combinations():
    corners = np.stack([np.arange(351), np.zeros(351)], axis=1)
    confs = np.ones(351)

    _, _, edge_coords, edge_mask, edge_ids = get_infer_edge_pairs(corners, confs)

    assert edge_coords.shape == (1, 61425, 2, 2)
    assert edge_mask.shape == (1, 61425)
    assert edge_ids.shape == (61425, 2)
