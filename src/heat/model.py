from typing import Any, Final

import numpy as np
import numpy.typing as npt
import torch

from .datasets.data_utils import get_pixel_features
from .infer import get_results, postprocess_preds
from .models.corner_models import HeatCorner
from .models.edge_models import HeatEdge
from .models.resnet import ResNetBackbone


class HEAT:
    _device: Final[torch.device]
    _backbone: Final[torch.nn.Module]
    _corner_model: Final[torch.nn.Module]
    _edge_model: Final[torch.nn.Module]
    _checkpoint_args: Any

    def __init__(self, use_gpu: bool) -> None:
        self._backbone = ResNetBackbone()
        strides = self._backbone.strides
        num_channels = self._backbone.num_channels
        self._corner_model = HeatCorner(
            input_dim=128,
            hidden_dim=256,
            num_feature_levels=4,
            backbone_strides=strides,
            backbone_num_channels=num_channels,
        )
        self._edge_model = HeatEdge(
            input_dim=128,
            hidden_dim=256,
            num_feature_levels=4,
            backbone_strides=strides,
            backbone_num_channels=num_channels,
        )

        # Choose to infer on CPU or GPU
        if use_gpu:
            assert torch.cuda.is_available(), "CUDA is not available."
            self._device = torch.device("cuda:0")
        else:
            self._device = torch.device("cpu")

        self._backbone = torch.nn.DataParallel(self._backbone)
        self._corner_model = torch.nn.DataParallel(self._corner_model)
        self._edge_model = torch.nn.DataParallel(self._edge_model)

        self._backbone.to(self._device)
        self._corner_model.to(self._device)
        self._edge_model.to(self._device)

    def load_checkpoint(self, checkpoint_path: str) -> int | None:
        # 学習済みモデルの読み込み
        checkpoint = torch.load(
            checkpoint_path, map_location=self._device, weights_only=False
        )
        self._ckpt_args = checkpoint["args"]
        self._backbone.load_state_dict(checkpoint["backbone"])
        self._corner_model.load_state_dict(checkpoint["corner_model"])
        self._edge_model.load_state_dict(checkpoint["edge_model"])

        if hasattr(self._ckpt_args, "image_size"):
            return self._ckpt_args.image_size
        return None

    def infer(self, bgr_image: npt.NDArray[np.uint8], infer_times=3):
        """
        与えられたBGR画像に対して、事前学習済みモデルを用いてコーナーとエッジを予測する推論を実行します。
        このメソッドは入力画像を処理し、ニューラルネットワークモデルに通し、後処理を適用した後に予測されたコーナーとエッジを返します。

        :param bgr_image:  `np.uint8`型NumPy配列で表される、BGR色空間の入力画像。
        :param infer_times: 実行する推論パスの数。デフォルトは3。
        :return: 以下の要素を含むタプル:
            - `pred_corners` (np.ndarray): 予測されたコーナーポイント。
            - `pos_edges` (np.ndarray): コーナー間の予測されたエッジ。
        """
        self._backbone.eval()
        self._corner_model.eval()
        self._edge_model.eval()

        image_size = (
            self._ckpt_args.image_size
            if hasattr(self._ckpt_args, "image_size")
            else 256
        )

        X = bgr_image.transpose(2, 0, 1) / 255.0
        X = torch.from_numpy(X[np.newaxis, :, :, :]).to(
            dtype=torch.float, device=self._device
        )

        pixels, pixel_features = get_pixel_features(image_size=image_size)

        with torch.inference_mode():
            pred_corners, pred_confs, pos_edges, _, _ = get_results(
                X,
                self._backbone,
                self._corner_model,
                self._edge_model,
                pixels,
                pixel_features,
                self._ckpt_args,
                infer_times,
                corner_thresh=0.01,
                image_size=image_size,
            )

        if pred_confs.shape[0] == 0:
            pred_confs = None

        pred_corners, _, pos_edges = postprocess_preds(
            pred_corners, pred_confs, pos_edges
        )

        return pred_corners, pos_edges
