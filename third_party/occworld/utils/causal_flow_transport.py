import torch
import torch.nn.functional as F


def dynamic_mask(semantics, dynamic_label_ids):
    mask = torch.zeros_like(semantics, dtype=torch.bool)
    for label_id in dynamic_label_ids:
        mask |= semantics == int(label_id)
    return mask


def splat_cumulative_dynamic_flow(
        source_semantics,
        cumulative_flow,
        dynamic_label_ids,
        point_cloud_range=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
        voxel_size=(0.4, 0.4, 0.4)):
    """Forward-splat source dynamic labels into future ego grids.

    cumulative_flow is measured in meters and maps source-grid coordinates
    directly to each target ego grid. Bilinear xy weights retain gradients
    with respect to flow; z uses nearest-voxel assignment.
    """
    if cumulative_flow.ndim != 6 or cumulative_flow.shape[-1] != 2:
        raise ValueError(
            f'Expected flow [B,T,X,Y,Z,2], got {cumulative_flow.shape}')
    batch_size, num_frames, size_x, size_y, size_z, _ = (
        cumulative_flow.shape)
    if source_semantics.shape != (batch_size, size_x, size_y, size_z):
        raise ValueError(
            f'Source shape {source_semantics.shape} does not match flow '
            f'{cumulative_flow.shape}')

    device = cumulative_flow.device
    dtype = cumulative_flow.dtype
    dynamic_ids = torch.as_tensor(
        dynamic_label_ids, device=device, dtype=torch.long)
    source_semantics = source_semantics.to(device=device, dtype=torch.long)
    source_dynamic = dynamic_mask(source_semantics, dynamic_label_ids)
    class_lookup = torch.full(
        (int(dynamic_ids.max().item()) + 1,),
        -1,
        dtype=torch.long,
        device=device)
    class_lookup[dynamic_ids] = torch.arange(len(dynamic_ids), device=device)
    lower = cumulative_flow.new_tensor(point_cloud_range[:3])
    voxel = cumulative_flow.new_tensor(voxel_size)
    flat_size = size_x * size_y * size_z

    batch_mass = []
    batch_support = []
    for batch_index in range(batch_size):
        source_indices = source_dynamic[batch_index].nonzero(as_tuple=False)
        if source_indices.numel() == 0:
            batch_mass.append(cumulative_flow.new_zeros(
                (num_frames, size_x, size_y, size_z, len(dynamic_ids))))
            batch_support.append(cumulative_flow.new_zeros(
                (num_frames, size_x, size_y, size_z)))
            continue
        source_points = lower + (
            source_indices.to(dtype) + 0.5) * voxel
        source_labels = source_semantics[batch_index][source_dynamic[batch_index]]
        source_classes = class_lookup[source_labels]
        source_values = F.one_hot(
            source_classes, num_classes=len(dynamic_ids)).to(dtype)

        frame_mass = []
        frame_support = []
        for frame_index in range(num_frames):
            moved = source_points.clone()
            moved[:, :2] += cumulative_flow[
                batch_index, frame_index][source_dynamic[batch_index]]
            continuous = (moved - lower) / voxel - 0.5
            base_xy = torch.floor(continuous[:, :2]).to(torch.long)
            fraction_xy = continuous[:, :2] - base_xy.to(dtype)
            target_z = torch.round(continuous[:, 2]).to(torch.long)
            class_accum = cumulative_flow.new_zeros(
                (flat_size, len(dynamic_ids)))
            support_accum = cumulative_flow.new_zeros(flat_size)

            for offset_x in (0, 1):
                for offset_y in (0, 1):
                    target_x = base_xy[:, 0] + offset_x
                    target_y = base_xy[:, 1] + offset_y
                    weight_x = (
                        fraction_xy[:, 0]
                        if offset_x else 1.0 - fraction_xy[:, 0])
                    weight_y = (
                        fraction_xy[:, 1]
                        if offset_y else 1.0 - fraction_xy[:, 1])
                    weight = weight_x * weight_y
                    valid = (
                        (target_x >= 0) & (target_x < size_x)
                        & (target_y >= 0) & (target_y < size_y)
                        & (target_z >= 0) & (target_z < size_z)
                    )
                    if not valid.any():
                        continue
                    flat_index = (
                        target_x[valid] * (size_y * size_z)
                        + target_y[valid] * size_z
                        + target_z[valid])
                    class_accum = class_accum.scatter_add(
                        0,
                        flat_index[:, None].expand(-1, len(dynamic_ids)),
                        source_values[valid] * weight[valid, None])
                    support_accum = support_accum.scatter_add(
                        0, flat_index, weight[valid])

            frame_mass.append(class_accum.reshape(
                size_x, size_y, size_z, len(dynamic_ids)))
            frame_support.append(support_accum.reshape(
                size_x, size_y, size_z))
        batch_mass.append(torch.stack(frame_mass, dim=0))
        batch_support.append(torch.stack(frame_support, dim=0))
    return torch.stack(batch_mass, dim=0), torch.stack(batch_support, dim=0)


def static_confidence_from_logits(baseline_logits, static_label_ids):
    static_ids = torch.as_tensor(
        static_label_ids, device=baseline_logits.device, dtype=torch.long)
    valid_static_ids = static_ids[static_ids < baseline_logits.shape[-1]]
    if valid_static_ids.numel() == 0:
        raise ValueError(
            'No static_label_ids are valid for baseline_logits with '
            f'{baseline_logits.shape[-1]} channels')
    static_logsumexp = torch.logsumexp(
        baseline_logits.index_select(-1, valid_static_ids), dim=-1)
    all_logsumexp = torch.logsumexp(baseline_logits, dim=-1)
    return torch.exp(static_logsumexp - all_logsumexp)


def transport_candidate_mask(
        baseline_prediction,
        support,
        baseline_logits=None,
        static_label_ids=(0, 1, 8, 11, 12, 13, 14, 15, 16),
        free_label=17,
        support_threshold=0.05,
        static_confidence_threshold=0.6,
        merge_mode='replace'):
    if merge_mode not in (
            'replace', 'union', 'free_only', 'static_confidence'):
        raise ValueError(
            'merge_mode must be one of "replace", "union", "free_only", or '
            f'"static_confidence", got {merge_mode}')
    if merge_mode == 'static_confidence' and baseline_logits is None:
        raise ValueError(
            'merge_mode="static_confidence" requires baseline_logits')
    if baseline_logits is not None and baseline_logits.shape[:-1] != (
            baseline_prediction.shape):
        raise ValueError(
            'baseline_logits shape must match baseline_prediction plus '
            f'class dim, got {baseline_logits.shape} vs '
            f'{baseline_prediction.shape}')

    transported = support > float(support_threshold)
    if merge_mode == 'free_only':
        transported &= baseline_prediction == int(free_label)
    elif merge_mode == 'static_confidence':
        static_confidence = static_confidence_from_logits(
            baseline_logits, static_label_ids)
        transported &= static_confidence < float(
            static_confidence_threshold)
    return transported


def merge_transport_with_baseline(
        baseline_prediction,
        class_mass,
        support,
        dynamic_label_ids,
        baseline_logits=None,
        static_label_ids=(0, 1, 8, 11, 12, 13, 14, 15, 16),
        free_label=17,
        support_threshold=0.05,
        static_confidence_threshold=0.6,
        merge_mode='replace',
        gate_prob=None,
        gate_threshold=0.5):
    if merge_mode not in (
            'replace', 'union', 'free_only', 'static_confidence'):
        raise ValueError(
            'merge_mode must be one of "replace", "union", "free_only", or '
            f'"static_confidence", got {merge_mode}')
    if merge_mode == 'static_confidence' and baseline_logits is None:
        raise ValueError(
            'merge_mode="static_confidence" requires baseline_logits')
    if baseline_logits is not None and baseline_logits.shape[:-1] != (
            baseline_prediction.shape):
        raise ValueError(
            'baseline_logits shape must match baseline_prediction plus '
            f'class dim, got {baseline_logits.shape} vs '
            f'{baseline_prediction.shape}')
    output = baseline_prediction.clone()
    if merge_mode == 'replace':
        baseline_dynamic = dynamic_mask(output, dynamic_label_ids)
        output[baseline_dynamic] = int(free_label)
    transported = transport_candidate_mask(
        output,
        support,
        baseline_logits=baseline_logits,
        static_label_ids=static_label_ids,
        free_label=free_label,
        support_threshold=support_threshold,
        static_confidence_threshold=static_confidence_threshold,
        merge_mode=merge_mode)
    if gate_prob is not None:
        if gate_prob.shape != support.shape:
            raise ValueError(
                f'gate_prob shape {gate_prob.shape} must match support '
                f'shape {support.shape}')
        transported &= gate_prob > float(gate_threshold)
    class_ids = torch.as_tensor(
        dynamic_label_ids, device=output.device, dtype=output.dtype)
    transported_labels = class_ids[class_mass.argmax(dim=-1)]
    output[transported] = transported_labels[transported]
    return output
