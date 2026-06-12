from pathlib import Path

import cv2
import numpy as np
import pytest

from heat.model import HEAT

pytestmark = pytest.mark.integration


class TestBatchInfer:
    @pytest.fixture
    def project_root(self) -> Path:
        return Path(__file__).parent.parent

    @pytest.fixture
    def test_data_dir(self, project_root: Path) -> Path:
        return project_root / "test_data"

    @pytest.fixture
    def checkpoint_file(self, project_root: Path, test_data_dir: Path) -> Path:
        checkpoint_file = project_root / "roof_edge_detection_parameter.pth"
        if not checkpoint_file.exists():
            checkpoint_file = test_data_dir / "roof_edge_detection_parameter.pth"
        return checkpoint_file

    @pytest.fixture
    def test_image(self, test_data_dir: Path) -> np.ndarray:
        image_path = test_data_dir / "test_image.png"
        return cv2.imread(str(image_path))

    @pytest.fixture
    def heat_model(self, checkpoint_file: Path) -> HEAT:
        model = HEAT()
        model.load_checkpoint(checkpoint_file)
        return model

    def test_infer_batch(self, heat_model: HEAT, test_image: np.ndarray):
        images = [test_image, test_image]
        results = heat_model.infer_batch(images)

        assert len(results) == 2
        for res in results:
            assert isinstance(res, tuple)
            assert len(res) == 2
            assert isinstance(res[0], np.ndarray)  # corners
            assert isinstance(res[1], np.ndarray)  # edges

        # Check if results for same image are identical
        np.testing.assert_array_equal(results[0][0], results[1][0])
        np.testing.assert_array_equal(results[0][1], results[1][1])

        # Compare with single infer
        single_res = heat_model.infer(test_image)
        np.testing.assert_array_equal(results[0][0], single_res[0])
        np.testing.assert_array_equal(results[0][1], single_res[1])
