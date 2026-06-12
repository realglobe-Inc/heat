# HEAT: Height Estimation and Attribute Transfer

<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white" width="9%" />
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

HEAT（Height Estimation and Attribute Transfer）は、画像からコーナーとエッジを検出するためのディープラーニングモデルです。

## 概要

このプロジェクトは、コンピュータビジョン技術を用いて画像内の構造的要素（コーナーとエッジ）を自動検出するためのPythonライブラリです。主な特徴：

- **Transformerベースのアーキテクチャ**: Deformable Transformerを使用した高精度なコーナー検出
- **エッジ予測**: 検出されたコーナー間のエッジ関係を予測
- **マルチスケール特徴抽出**: ResNetバックボーンによる効率的な特徴抽出
- **CUDA 13.x 対応**: 最新のCUDA環境でのビルドと動作をサポート
- **推論API**: 簡単に使用できるPython API

## アーキテクチャ

HEATモデルは以下の主要コンポーネントで構成されています：

1. **ResNetバックボーン**: 入力画像から多層特徴を抽出
2. **HeatCornerモデル**: Deformable Transformerを使用したコーナー検出
3. **HeatEdgeモデル**: コーナー間のエッジ関係を予測
4. **後処理**: Non-Maximum Suppression（NMS）による結果の精製

## インストール

### 必要な環境

- Python 3.13以上
- PyTorch 2.10.0以上
- CUDA対応GPU
  - CUDA 13.0 (推奨) または CUDA 12.x

### パッケージのインストール

```bash
poetry install
```

`poetry install` を実行すると、依存ライブラリのインストールに加えて、C++ 拡張（MultiScaleDeformableAttention）の自動ビルドが行われます。

> [!TIP]
> ビルド環境（CUDAのバージョン等）により C++ 拡張のビルドに失敗する場合でも、自動的に純粋な PyTorch 実装へとフォールバックするよう設計されているため、推論機能は引き続き利用可能です。

> [!IMPORTANT]
> ビルドには `torch` と `ninja` が必要です。これらは `pyproject.toml` の `build-system` および `dev-dependencies` に含まれています。
> CUDA 13.x 環境でのビルドもサポートされています。

### 利用時の注意

C++ 拡張を使用する際は、共有ライブラリのシンボル解決を正しく行うため、**必ず `import torch` を最初に行ってください。**

```python
import torch
import MultiScaleDeformableAttention
```

## 使用方法

### Python API

```python
import cv2
import numpy as np
from heat.model import HEAT

# モデルの初期化
model = HEAT(use_gpu=True)  # GPUを使用する場合
model.load_checkpoint("path/to/checkpoint.pth")

# 画像の読み込み（BGR形式）
image = cv2.imread("path/to/image.jpg")

# 推論の実行
pred_corners, pos_edges = model.infer(image, infer_times=3)

print(f"検出されたコーナー数: {len(pred_corners)}")
print(f"検出されたエッジ数: {len(pos_edges)}")
```

### 推論パラメータ

- `infer_times`: 推論パスの回数（デフォルト: 3）
- `corner_thresh`: コーナー検出の閾値（デフォルト: 0.01）
- `image_size`: 処理画像サイズ（チェックポイントから自動設定）

## プロジェクト構造

```
src/heat/
├── __init__.py
├── model.py              # メインのHEATモデルクラス
├── infer.py             # 推論処理とユーティリティ
├── datasets/            # データ処理ユーティリティ
├── models/              # ニューラルネットワークモデル
│   ├── corner_models.py # コーナー検出モデル
│   ├── edge_models.py   # エッジ予測モデル
│   ├── resnet.py        # ResNetバックボーン
│   ├── deformable_transformer.py  # Deformable Transformer
│   └── ops/             # カスタムオペレーション
└── utils/               # ユーティリティ関数
```

## 主要機能

### コーナー検出

HeatCornerモデルは以下の機能を提供します：

- マルチスケール特徴抽出
- Deformable Transformerによる位置エンコーディング
- 高精度なコーナー位置予測
- Non-Maximum Suppressionによる重複除去

### エッジ予測

HeatEdgeモデルは検出されたコーナー間の関係を分析し：

- コーナーペア間のエッジ存在確率を計算
- 複数回の推論による結果の精製
- 構造的整合性の確保

### 推論プロセス

1. **前処理**: 入力画像の正規化とリサイズ
2. **特徴抽出**: ResNetバックボーンによる多層特徴抽出
3. **コーナー検出**: Transformerベースのコーナー予測
4. **エッジ予測**: コーナー間の関係性分析
5. **後処理**: 結果の精製と構造化

## 依存関係

主要な依存パッケージ：

- `torch`: PyTorchディープラーニングフレームワーク (2.10.0以上)
- `torchvision`: コンピュータビジョンユーティリティ (0.25.0以上)
- `opencv-python`: 画像処理
- `numpy`: 数値計算
- `einops`: テンソル操作
- `scipy`: 科学計算
- `matplotlib`: 可視化

## 開発

### 開発環境のセットアップ

```bash
poetry install
```

### コードフォーマット

```bash
poetry run ruff format .
```

## ライセンス

このプロジェクトはGPL v3ライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 技術的詳細

### Deformable Attention

このプロジェクトでは、効率的な注意機構として Multi-Scale Deformable Attention を使用しています。これにより：

- 計算効率の向上
- 長距離依存関係の効果的な捕捉
- マルチスケール特徴の統合

また、CUDA 13環境における数学関数の競合や警告に対応しており、最新のハードウェア環境でも安定して動作します。

### モデルアーキテクチャ

- **入力**: BGR画像（任意サイズ、内部で256x256にリサイズ）
- **出力**: コーナー座標とエッジ接続情報
- **バックボーン**: ResNet-50ベースの特徴抽出器
- **デコーダー**: Deformable Transformerデコーダー
