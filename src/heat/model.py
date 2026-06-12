from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from numpy.typing import NDArray

from heat.datasets.data_utils import get_pixel_features
from heat.infer import get_results_batch, postprocess_predicates
from heat.models.corner_models import HeatCorner
from heat.models.edge_models import HeatEdge
from heat.models.resnet import ResNetBackbone


class HEAT:
    _device: Final[torch.device]
    _backbone: Final[torch.nn.Module]
    _corner_model: Final[torch.nn.Module]
    _edge_model: Final[torch.nn.Module]
    _checkpoint_args: Any
    _pixels: NDArray[Any] | None = None
    _pixel_features: torch.Tensor | None = None
    _cached_image_size: int | None = None

    def __init__(self, force_cpu: bool = False):
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
        self._device = (
            torch.device("cpu")
            if force_cpu or not torch.cuda.is_available()
            else torch.device("cuda")
        )

        self._backbone = torch.nn.DataParallel(self._backbone)
        self._corner_model = torch.nn.DataParallel(self._corner_model)
        self._edge_model = torch.nn.DataParallel(self._edge_model)

        self._backbone.to(self._device)
        self._corner_model.to(self._device)
        self._edge_model.to(self._device)

    @property
    def device(self) -> torch.device:
        """
        モデルが現在使用しているデバイスを取得します。
        """
        return self._device

    def load_checkpoint(self, checkpoint_path: Path) -> int | None:
        # 学習済みモデルの読み込み
        checkpoint = torch.load(
            checkpoint_path, map_location=self._device, weights_only=False
        )
        self._checkpoint_args = checkpoint["args"]
        self._backbone.load_state_dict(checkpoint["backbone"])
        self._corner_model.load_state_dict(checkpoint["corner_model"])
        self._edge_model.load_state_dict(checkpoint["edge_model"])

        if hasattr(self._checkpoint_args, "image_size"):
            return self._checkpoint_args.image_size
        return None

    def infer(
        self, bgr_image: NDArray[np.uint8], infer_times: int = 3
    ) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
        """
        与えられたBGR画像に対して、事前学習済みモデルを用いてコーナーとエッジを予測する推論を実行します。
        このメソッドは入力画像を処理し、ニューラルネットワークモデルに通し、後処理を適用した後に予測されたコーナーとエッジを返します。

        :param bgr_image:  `np.uint8`型NumPy配列で表される、BGR色空間の入力画像。
        :param infer_times: 実行する推論パスの数。デフォルトは3。
        :returns: 以下の要素を含むタプル:
            - `pred_corners` (np.ndarray): 予測されたコーナーポイント。
            - `pos_edges` (np.ndarray): コーナー間の予測されたエッジ。
        """
        results = self.infer_batch([bgr_image], infer_times)
        return results[0]

    def infer_batch(
        self, bgr_images: list[NDArray[np.uint8]], infer_times: int = 3
    ) -> list[tuple[NDArray[np.float64], NDArray[np.int32]]]:
        """
        与えられた複数のBGR画像に対して、推論を一括実行します。

        :param bgr_images: `np.uint8`型NumPy配列のリスト。
        :param infer_times: 各画像に対して実行する推論パスの数。デフォルトは3。
        :returns: 各画像に対する (pred_corners, pos_edges) のタプルのリスト。
        """
        self._backbone.eval()
        self._corner_model.eval()
        self._edge_model.eval()

        image_size = (
            self._checkpoint_args.image_size
            if hasattr(self._checkpoint_args, "image_size")
            else 256
        )

        if (
            self._cached_image_size != image_size
            or self._pixels is None
            or self._pixel_features is None
        ):
            self._pixels, self._pixel_features = get_pixel_features(
                image_size=image_size
            )
            self._pixel_features = self._pixel_features.to(self._device)
            self._cached_image_size = image_size

        # すべての画像を1つのバッチにまとめる
        x_list = []
        for bgr_image in bgr_images:
            x = bgr_image.transpose(2, 0, 1) / 255.0
            x_list.append(torch.from_numpy(x).to(dtype=torch.float))
        x_batch = torch.stack(x_list).to(device=self._device)

        with torch.inference_mode():
            # バックボーンの計算を一括で行う
            xs, mask, all_feats = self._backbone(x_batch)

            # バッチ推論の実行
            results = get_results_batch(
                x_batch,
                None,
                self._corner_model,
                self._edge_model,
                self._pixels,
                self._pixel_features,
                self._checkpoint_args,
                infer_times,
                corner_thresh=0.01,
                image_size=image_size,
                backbone_results=(xs, mask, all_feats),
            )

            batch_results = []
            for i in range(len(bgr_images)):
                pred_corners, pred_confs, pos_edges, edge_confs, c_outputs_np = results[
                    i
                ]

                if pred_confs.shape[0] == 0:
                    pred_confs = None

                pred_corners, _, pos_edges = postprocess_predicates(
                    pred_corners, pred_confs, pos_edges
                )
                batch_results.append((pred_corners, pos_edges))

        return batch_results
