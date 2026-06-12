import urllib
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

from heat.model import HEAT


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
                [18, 25],
                [0, 2],
                [5, 8],
                [5, 10],
                [33, 35],
                [24, 28],
                [9, 11],
                [14, 22],
                [9, 14],
                [11, 22],
                [6, 9],
                [11, 24],
                [19, 25],
                [11, 29],
                [1, 4],
                [3, 8],
                [3, 10],
                [15, 23],
                [6, 21],
                [25, 33],
                [12, 13],
                [10, 12],
                [25, 35],
                [12, 16],
                [26, 36],
                [7, 10],
                [21, 28],
                [28, 31],
                [4, 6],
                [4, 5],
                [2, 4],
                [4, 9],
                [17, 20],
                [17, 21],
                [22, 27],
                [30, 32],
                [30, 33],
                [8, 10],
                [23, 25],
                [18, 19],
                [23, 30],
                [32, 33],
                [32, 34],
                [5, 6],
            ]
        )

        np.testing.assert_array_equal(actual_corners, expected_corners)
        np.testing.assert_array_equal(actual_edges, expected_edges)
