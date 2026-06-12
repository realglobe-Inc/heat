import urllib
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

from heat.model import HEAT

pytestmark = pytest.mark.integration


class TestRunIntegration:
    """runコマンドの統合テスト"""

    @pytest.fixture
    def project_root(self) -> Path:
        """プロジェクトルートを取得"""
        return Path(__file__).parent.parent

    @pytest.fixture
    def test_data_dir(self, project_root: Path) -> Path:
        """テストデータディレクトリを取得"""
        return project_root / "test_data"

    @pytest.fixture
    def checkpoint_file(self, project_root: Path, test_data_dir: Path) -> Path:
        """チェックポイントファイルを用意"""
        checkpoint_file = project_root / "roof_edge_detection_parameter.pth"
        if checkpoint_file.exists():
            return checkpoint_file

        checkpoint_file = test_data_dir / "roof_edge_detection_parameter.pth"
        if checkpoint_file.exists():
            return checkpoint_file

        url = "https://github.com/realglobe-Inc/bldg-lod2-tool/releases/download/PretrainedModels-1.0/roof_edge_detection_parameter.pth"
        urllib.request.urlretrieve(url, checkpoint_file)
        return checkpoint_file

    @pytest.fixture
    def test_image(self, test_data_dir: Path) -> np.ndarray:
        """テスト用画像を読み込む"""
        image_path = test_data_dir / "test_image.png"
        return cv2.imread(str(image_path))

    @pytest.fixture
    def heat_model(self, checkpoint_file: Path) -> HEAT:
        """HEATモデルを初期化"""
        model = HEAT()
        model.load_checkpoint(checkpoint_file)
        return model

    def test_infer(self, heat_model: HEAT, test_image: np.ndarray):
        """inferメソッドのテスト"""
        actual_corners, actual_edges = heat_model.infer(test_image)
        expected_corners = np.array(
            [
                [39, 44],
                [70, 59],
                [45, 61],
                [27, 69],
                [65, 70],
                [53, 72],
                [84, 85],
                [8, 88],
                [37, 93],
                [116, 96],
                [35, 101],
                [99, 106],
                [31, 108],
                [15, 111],
                [154, 112],
                [179, 112],
                [25, 117],
                [49, 120],
                [193, 120],
                [191, 126],
                [45, 127],
                [64, 128],
                [147, 128],
                [166, 132],
                [108, 133],
                [183, 141],
                [223, 142],
                [140, 143],
                [99, 144],
                [109, 145],
                [156, 153],
                [105, 154],
                [166, 158],
                [174, 162],
                [153, 174],
                [165, 180],
                [197, 196],
            ]
        )
        expected_edges = np.array(
            [
                [0, 2],
                [1, 4],
                [2, 4],
                [3, 8],
                [3, 10],
                [4, 5],
                [4, 6],
                [4, 9],
                [5, 6],
                [5, 8],
                [5, 10],
                [6, 9],
                [6, 21],
                [7, 10],
                [8, 10],
                [9, 11],
                [9, 14],
                [10, 12],
                [11, 22],
                [11, 24],
                [11, 29],
                [12, 13],
                [12, 16],
                [14, 22],
                [15, 23],
                [17, 20],
                [17, 21],
                [18, 19],
                [18, 25],
                [19, 25],
                [21, 28],
                [22, 27],
                [23, 25],
                [23, 30],
                [24, 28],
                [25, 33],
                [25, 35],
                [26, 36],
                [28, 31],
                [30, 32],
                [30, 33],
                [32, 33],
                [32, 34],
                [33, 35],
            ]
        )

        np.testing.assert_array_equal(actual_corners, expected_corners)
        np.testing.assert_array_equal(actual_edges, expected_edges)
