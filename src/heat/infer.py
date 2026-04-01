from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from scipy.ndimage import maximum_filter, minimum_filter

from heat.models.corner_to_edge import get_infer_edge_pairs


def corner_nms(predicates, confs, image_size):
    data = np.zeros([image_size, image_size])
    neighborhood_size = 5
    threshold = 0

    for i in range(len(predicates)):
        data[predicates[i, 1], predicates[i, 0]] = confs[i]

    data_max = maximum_filter(data, neighborhood_size)
    maxima = data == data_max
    data_min = minimum_filter(data, neighborhood_size)
    diff = (data_max - data_min) > threshold
    maxima[diff == 0] = 0

    results = np.where(maxima > 0)
    filtered_predicates = np.stack([results[1], results[0]], axis=-1)

    new_confs = []
    for pred in filtered_predicates:
        new_confs.append(data[pred[1], pred[0]])
    new_confs = np.array(new_confs)

    return filtered_predicates, new_confs


def get_results(
    image,
    backbone,
    corner_model,
    edge_model,
    pixels,
    pixel_features,
    args,
    infer_times,
    corner_thresh=0.5,
    image_size=256,
    backbone_results=None,
):
    if backbone_results is not None:
        image_feats, feat_mask, all_image_feats = backbone_results
    else:
        image_feats, feat_mask, all_image_feats = backbone(image)
    pixel_features = pixel_features.unsqueeze(0).expand(image.shape[0], -1, -1, -1)
    predicates_s1 = corner_model(
        image_feats, feat_mask, pixel_features, pixels, all_image_feats
    )

    c_outputs = predicates_s1
    # get predicted corners
    c_outputs_np = c_outputs[0].detach().cpu().numpy()
    pos_indices = np.where(c_outputs_np >= corner_thresh)
    pred_corners = pixels[pos_indices]
    pred_confs = c_outputs_np[pos_indices]
    pred_corners, pred_confs = corner_nms(
        pred_corners, pred_confs, image_size=c_outputs.shape[1]
    )

    if len(pred_corners) < 2:
        return (
            pred_corners,
            pred_confs,
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float64),
            c_outputs_np,
        )

    pred_corners, pred_confs, edge_coords, edge_mask, edge_ids = get_infer_edge_pairs(
        pred_corners, pred_confs
    )
    edge_coords = edge_coords.to(image.device)
    edge_mask = edge_mask.to(image.device)

    corner_nums = torch.tensor([len(pred_corners)]).to(image.device)
    max_candidates = torch.stack(
        [corner_nums.max() * args.corner_to_edge_multiplier] * len(corner_nums), dim=0
    )

    all_pos_ids = set()
    all_edge_confs = {}

    for tt in range(infer_times):
        if tt == 0:
            gt_values = torch.zeros_like(edge_mask).to(image.device).long()
            gt_values[:, :] = 2

        # run the edge model
        s1_logits, s2_logits_hb, s2_logits_rel, selected_ids, s2_mask, s2_gt_values = (
            edge_model(
                image_feats,
                feat_mask,
                pixel_features,
                edge_coords,
                edge_mask,
                gt_values,
                corner_nums,
                max_candidates,
                True,
            )
        )

        num_total = s1_logits.shape[2]
        num_selected = selected_ids.shape[1]
        num_filtered = num_total - num_selected

        s2_predicates_hb = s2_logits_hb.squeeze(0).softmax(0)
        s2_predicates_hb_np = s2_predicates_hb[1, :].detach().cpu().numpy()

        selected_ids = selected_ids.squeeze(0).detach().cpu().numpy()
        if tt != infer_times - 1:
            s2_predicates_np = s2_predicates_hb_np

            pos_edge_ids = np.where(s2_predicates_np >= 0.3)
            neg_edge_ids = np.where(s2_predicates_np <= 0.01)
            for pos_id in pos_edge_ids[0]:
                actual_id = selected_ids[pos_id]
                if gt_values[0, actual_id] != 2:
                    continue
                all_pos_ids.add(actual_id)
                all_edge_confs[actual_id] = s2_predicates_np[pos_id]
                gt_values[0, actual_id] = 1
            for neg_id in neg_edge_ids[0]:
                actual_id = selected_ids[neg_id]
                if gt_values[0, actual_id] != 2:
                    continue
                gt_values[0, actual_id] = 0
            num_to_pred = (gt_values == 2).sum()
            if num_to_pred <= num_filtered:
                break
        else:
            s2_predicates_np = s2_predicates_hb_np

            pos_edge_ids = np.where(s2_predicates_np >= 0.2)
            for pos_id in pos_edge_ids[0]:
                actual_id = selected_ids[pos_id]
                if s2_mask[0][pos_id] is True or gt_values[0, actual_id] != 2:
                    continue
                all_pos_ids.add(actual_id)
                all_edge_confs[actual_id] = s2_predicates_np[pos_id]

    # print('Inference time {}'.format(tt+1))
    pos_edge_ids = list(all_pos_ids)
    edge_confs = [all_edge_confs[idx] for idx in pos_edge_ids]
    pos_edges = edge_ids[pos_edge_ids].cpu().numpy()
    edge_confs = np.array(edge_confs)

    if image_size != 256:
        pred_corners = pred_corners / (image_size / 256)

    return pred_corners, pred_confs, pos_edges, edge_confs, c_outputs_np


def postprocess_predicates(corners, confs, edges):
    corner_degrees = {}
    for edge_pair in edges:
        corner_degrees[edge_pair[0]] = corner_degrees.setdefault(edge_pair[0], 0) + 1
        corner_degrees[edge_pair[1]] = corner_degrees.setdefault(edge_pair[1], 0) + 1
    good_ids = [i for i in range(len(corners)) if i in corner_degrees]
    if len(good_ids) == len(corners):
        return corners, confs, edges
    else:
        good_corners = corners[good_ids]
        good_confs = confs[good_ids]
        id_mapping = {value: idx for idx, value in enumerate(good_ids)}
        new_edges = []
        for edge_pair in edges:
            new_pair = (id_mapping[edge_pair[0]], id_mapping[edge_pair[1]])
            new_edges.append(new_pair)
        new_edges = np.array(new_edges)
        return good_corners, good_confs, new_edges


def get_results_batch(
    images,
    backbone,
    corner_model,
    edge_model,
    pixels,
    pixel_features,
    args,
    infer_times,
    corner_thresh=0.5,
    image_size=256,
    backbone_results=None,
):
    """
    複数の画像に対してコーナーとエッジの予測結果を一括で取得します。

    :param images: (B, C, H, W) の画像テンソル。
    :param backbone: バックボーンモデル。
    :param corner_model: コーナー検出モデル。
    :param edge_model: エッジ検出モデル。
    :param pixels: 画素座標のテンソル。
    :param pixel_features: 画素特徴量のテンソル。
    :param args: 推論用引数（`corner_to_edge_multiplier` を含む）。
    :param infer_times: エッジ推論の反復回数。
    :param corner_thresh: コーナー検出のしきい値。デフォルトは 0.5。
    :param image_size: 入力画像のサイズ。デフォルトは 256。
    :param backbone_results: 事前に計算されたバックボーンの結果。
    :returns: 各画像に対する
        (pred_corners, pred_confs, pos_edges, edge_confs, c_outputs_np) のリスト。
    """
    if backbone_results is not None:
        image_feats, feat_mask, all_image_feats = backbone_results
    else:
        image_feats, feat_mask, all_image_feats = backbone(images)

    batch_size = images.shape[0]
    device = images.device

    pixel_features_expanded = pixel_features.unsqueeze(0).expand(batch_size, -1, -1, -1)
    c_outputs = corner_model(
        image_feats, feat_mask, pixel_features_expanded, pixels, all_image_feats
    )

    def process_one_corner(i):
        c_out_np = c_outputs[i].detach().cpu().numpy()
        pos_indices = np.where(c_out_np >= corner_thresh)
        pred_corners = pixels[pos_indices]
        pred_confs = c_out_np[pos_indices]
        pred_corners, pred_confs = corner_nms(
            pred_corners, pred_confs, image_size=c_outputs.shape[1]
        )
        if len(pred_corners) < 2:
            return i, pred_corners, pred_confs, None, None, None, c_out_np

        pred_corners, pred_confs, edge_coords, edge_mask, edge_ids = (
            get_infer_edge_pairs(pred_corners, pred_confs)
        )
        return i, pred_corners, pred_confs, edge_coords, edge_mask, edge_ids, c_out_np

    with ThreadPoolExecutor() as executor:
        cpu_results = list(executor.map(process_one_corner, range(batch_size)))

    final_pred_corners = [None] * batch_size
    final_pred_confs = [None] * batch_size
    final_c_outputs_np = [None] * batch_size

    batch_edge_coords_list = []
    batch_edge_mask_list = []
    batch_edge_ids_list = []
    batch_corner_nums_list = []

    valid_batch_indices = []

    for i, corners, confs, e_coords, e_mask, e_ids, c_out_np in cpu_results:
        final_pred_corners[i] = corners
        final_pred_confs[i] = confs
        final_c_outputs_np[i] = c_out_np
        if e_coords is not None:
            valid_batch_indices.append(i)
            batch_edge_coords_list.append(e_coords.squeeze(0))
            batch_edge_mask_list.append(e_mask.squeeze(0))
            batch_edge_ids_list.append(e_ids)
            batch_corner_nums_list.append(len(corners))
        else:
            batch_corner_nums_list.append(len(corners))

    if not valid_batch_indices:
        results = []
        for i in range(batch_size):
            pc = final_pred_corners[i]
            if image_size != 256:
                pc = pc / (image_size / 256)
            results.append(
                (
                    pc,
                    final_pred_confs[i],
                    np.array([], dtype=np.int32),
                    np.array([], dtype=np.float64),
                    final_c_outputs_np[i],
                )
            )
        return results

    max_edges = max(len(m) for m in batch_edge_mask_list)
    padded_edge_coords = torch.zeros(
        (len(valid_batch_indices), max_edges, 2, 2), dtype=torch.long, device=device
    )
    padded_edge_mask = torch.ones(
        (len(valid_batch_indices), max_edges), dtype=torch.bool, device=device
    )

    for idx, _ in enumerate(valid_batch_indices):
        n = len(batch_edge_mask_list[idx])
        padded_edge_coords[idx, :n] = batch_edge_coords_list[idx]
        padded_edge_mask[idx, :n] = False

    corner_nums = torch.tensor(batch_corner_nums_list, device=device)[
        valid_batch_indices
    ]
    max_candidates = (corner_nums * args.corner_to_edge_multiplier).long()

    v_image_feats = {k: v[valid_batch_indices] for k, v in image_feats.items()}
    v_feat_mask = feat_mask[valid_batch_indices]
    v_pixel_features = pixel_features_expanded[valid_batch_indices]

    all_pos_ids = [set() for _ in range(len(valid_batch_indices))]
    all_edge_confs = [{} for _ in range(len(valid_batch_indices))]
    gt_values = torch.full(padded_edge_mask.shape, 2, dtype=torch.long, device=device)

    for tt in range(infer_times):
        s1_logits, s2_logits_hb, s2_logits_rel, selected_ids, s2_mask, s2_gt_values = (
            edge_model(
                v_image_feats,
                v_feat_mask,
                v_pixel_features,
                padded_edge_coords,
                padded_edge_mask,
                gt_values,
                corner_nums,
                max_candidates,
                True,
            )
        )

        s2_predicates_hb = s2_logits_hb.softmax(1)
        s2_predicates_hb_np = s2_predicates_hb[:, 1, :].detach().cpu().numpy()
        selected_ids_np = selected_ids.detach().cpu().numpy()
        s2_mask_np = s2_mask.detach().cpu().numpy()

        for idx in range(len(valid_batch_indices)):
            s2_pred_np = s2_predicates_hb_np[idx]
            sel_ids = selected_ids_np[idx]

            if tt != infer_times - 1:
                pos_edge_ids = np.where(s2_pred_np >= 0.3)[0]
                neg_edge_ids = np.where(s2_pred_np <= 0.01)[0]
                for pos_id in pos_edge_ids:
                    actual_id = sel_ids[pos_id]
                    if gt_values[idx, actual_id] != 2:
                        continue
                    all_pos_ids[idx].add(actual_id)
                    all_edge_confs[idx][actual_id] = s2_pred_np[pos_id]
                    gt_values[idx, actual_id] = 1
                for neg_id in neg_edge_ids:
                    actual_id = sel_ids[neg_id]
                    if gt_values[idx, actual_id] != 2:
                        continue
                    gt_values[idx, actual_id] = 0
            else:
                pos_edge_ids = np.where(s2_pred_np >= 0.2)[0]
                for pos_id in pos_edge_ids:
                    actual_id = sel_ids[pos_id]
                    if (
                        s2_mask_np[idx][pos_id] is True
                        or gt_values[idx, actual_id] != 2
                    ):
                        continue
                    all_pos_ids[idx].add(actual_id)
                    all_edge_confs[idx][actual_id] = s2_pred_np[pos_id]

    batch_results = [None] * batch_size
    v_idx = 0
    for i in range(batch_size):
        if i in valid_batch_indices:
            p_edge_ids = list(all_pos_ids[v_idx])
            e_confs = np.array([all_edge_confs[v_idx][idx] for idx in p_edge_ids])
            p_edges = batch_edge_ids_list[v_idx][p_edge_ids].cpu().numpy()

            pred_c = final_pred_corners[i]
            if image_size != 256:
                pred_c = pred_c / (image_size / 256)

            batch_results[i] = (
                pred_c,
                final_pred_confs[i],
                p_edges,
                e_confs,
                final_c_outputs_np[i],
            )
            v_idx += 1
        else:
            pred_c = final_pred_corners[i]
            if image_size != 256:
                pred_c = pred_c / (image_size / 256)
            batch_results[i] = (
                pred_c,
                final_pred_confs[i],
                np.array([], dtype=np.int32),
                np.array([], dtype=np.float64),
                final_c_outputs_np[i],
            )

    return batch_results
