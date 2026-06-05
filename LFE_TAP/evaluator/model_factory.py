import os

from LFE_TAP.evaluator.prediction import TAPFormerCowDense_online, TAPFormer_online


_TAPFORMER_MODEL_NAMES = {"tapformer", "tapformer_online"}
_COW_DENSE_MODEL_NAMES = {"tapformer_cow_dense", "tapformer_cow_dense_online", "cow_dense"}


def _get_common_model_kwargs(model_cfg):
    return {
        "window_size": int(model_cfg.get("window_size", 16)),
        "stride": int(model_cfg.get("stride", 4)),
        "corr_radius": int(model_cfg.get("corr_radius", 3)),
        "corr_levels": int(model_cfg.get("corr_levels", 3)),
        "backbone": model_cfg.get("backbone", "basic"),
        "hidden_size": int(model_cfg.get("hidden_size", 384)),
        "space_depth": int(model_cfg.get("space_depth", 3)),
        "time_depth": int(model_cfg.get("time_depth", 3)),
    }


def _get_cow_model_kwargs(model_cfg):
    return {
        "cow_refine_model": str(model_cfg.get("cow_refine_model", "vits")),
        "cow_refine_patch_size": int(model_cfg.get("cow_refine_patch_size", 4)),
        "cow_refine_blocks": model_cfg.get("cow_refine_blocks", None),
        "cow_temporal_interleave_stride": int(model_cfg.get("cow_temporal_interleave_stride", 2)),
        "cow_tracking_down_ratio": int(model_cfg.get("cow_tracking_down_ratio", 2)),
        "cow_limit_flow": bool(model_cfg.get("cow_limit_flow", True)),
        "cow_max_flow_update_ratio": float(model_cfg.get("cow_max_flow_update_ratio", 0.15)),
        "cow_max_flow_magnitude_ratio": float(model_cfg.get("cow_max_flow_magnitude_ratio", 1.0)),
        "cow_refine_checkpoint": bool(model_cfg.get("cow_refine_checkpoint", False)),
        "cow_info_update_mode": str(model_cfg.get("cow_info_update_mode", "direct")),
        "cow_online_use_window_init": bool(model_cfg.get("cow_online_use_window_init", False)),
        "cow_online_use_global_first_anchor": bool(model_cfg.get("cow_online_use_global_first_anchor", False)),
        "cow_online_use_memory_features": bool(model_cfg.get("cow_online_use_memory_features", False)),
        "cow_online_num_memory_frames": int(model_cfg.get("cow_online_num_memory_frames", 10)),
        "cow_frontend_type": str(model_cfg.get("cow_frontend_type", "base")),
        "cow_anchor_state_mix": float(model_cfg.get("cow_anchor_state_mix", 0.7)),
        "cow_anchor_skip_mix": float(model_cfg.get("cow_anchor_skip_mix", 0.7)),
    }


def build_eval_model_from_config(model_cfg, inference_mode="online"):
    if str(inference_mode).lower().strip() != "online":
        raise ValueError("Only online inference_mode is supported for real-dataset evaluation.")

    model_name = str(
        model_cfg.get("model_name", model_cfg.get("name", "tapformer_online"))
    ).lower().strip()
    common_kwargs = _get_common_model_kwargs(model_cfg)

    if model_name in _TAPFORMER_MODEL_NAMES:
        return TAPFormer_online(**common_kwargs)

    if model_name in _COW_DENSE_MODEL_NAMES:
        cow_kwargs = _get_cow_model_kwargs(model_cfg)
        cow_kwargs.update(common_kwargs)
        return TAPFormerCowDense_online(**cow_kwargs)

    raise ValueError(
        f"Unsupported model_name={model_name}. "
        "Use one of: tapformer, tapformer_online, tapformer_cow_dense, tapformer_cow_dense_online, cow_dense."
    )
