import importlib
import sys
from pathlib import Path

import torch
import yaml

from LFE_TAP.evaluator.evaluation_pred import CowTrackerEvaluationPredictor, EvaluationPredictor
from LFE_TAP.evaluator.prediction import TAPFormerCowDense_online, TAPFormer_online


_TAPFORMER_MODEL_NAMES = {"tapformer", "tapformer_online"}
_COW_DENSE_MODEL_NAMES = {"tapformer_cow_dense", "tapformer_cow_dense_online", "cow_dense"}
_TAPFORMER_BACKEND_NAMES = {"tapformer", "tapformer_family", "internal"}
_COWTRACKER_BACKEND_NAMES = {"cowtracker"}
_FREEZE_CONFIG_KEYS = {
    "freeze_vggt",
    "freeze_aggregator",
    "freeze_feature_extractor",
    "freeze_tracking_head",
}


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


def _default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_device(device_value=None):
    if device_value is None:
        return _default_device()
    return torch.device(str(device_value))


def _normalize_eval_backend(backend_name):
    backend = str(backend_name or "tapformer_family").lower().strip()
    if backend in _TAPFORMER_BACKEND_NAMES:
        return "tapformer_family"
    if backend in _COWTRACKER_BACKEND_NAMES:
        return "cowtracker"
    raise ValueError(
        f"Unsupported eval_model.backend={backend_name}. "
        "Use one of: tapformer_family, cowtracker."
    )


def _get_eval_model_config(eval_cfg):
    eval_model_cfg = dict(eval_cfg.get("eval_model", {}))
    eval_model_cfg["backend"] = _normalize_eval_backend(
        eval_model_cfg.get("backend", "tapformer_family")
    )
    return eval_model_cfg


def _get_config_base_dir(cfg):
    config_path = cfg.get("__config_path__")
    if config_path is not None:
        return Path(config_path).expanduser().resolve().parent
    return Path.cwd()


def _resolve_path(path_value, base_dir):
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (Path(base_dir) / path).resolve()
    return path


def _load_yaml_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        return ckpt
    raise ValueError("Checkpoint must be a state_dict-like dict.")


def _normalize_state_dict_keys(state_dict):
    if state_dict and all(k.startswith("module.") for k in state_dict):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def _model_uses_vggt(cfg):
    model_cfg = cfg.get("model", {})
    return str(model_cfg.get("backbone_type", "vggt")).lower().strip() == "vggt"


def _ensure_vggt_available(repo_root):
    vggt_root = Path(repo_root) / "cowtracker" / "thirdparty" / "vggt"
    required_files = [
        vggt_root / "vggt" / "models" / "aggregator.py",
        vggt_root / "vggt" / "heads" / "dpt_head.py",
    ]
    missing = [path for path in required_files if not path.is_file()]
    if missing:
        missing_rel = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "VGGT submodule is required for CoWTracker evaluation, but files are missing:\n"
            f"{missing_rel}\n"
            "Run: git submodule update --init --recursive in the CowTracker repo."
        )


def _get_cowtracker_model_kwargs(cfg):
    model_cfg = dict(cfg.get("model", {}))
    for key in _FREEZE_CONFIG_KEYS:
        model_cfg.pop(key, None)
    return model_cfg


def _autocast_dtype(precision):
    precision = str(precision).lower().strip()
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision in {"fp32", "32", "float32", "none"}:
        return None
    raise ValueError("precision must be one of: bf16, fp16, fp32")


def _get_eval_resolution(eval_cfg):
    eval_resolution = eval_cfg.get("eval_resolution")
    if eval_resolution is None:
        return None
    if len(eval_resolution) != 2:
        raise ValueError("eval_resolution must be a 2-element [height, width] list.")
    return tuple(int(v) for v in eval_resolution)


def _load_tapformer_checkpoint(eval_cfg, device):
    ckpt_root = _resolve_path(eval_cfg["ckpt_root"], _get_config_base_dir(eval_cfg))
    state_dict = torch.load(str(ckpt_root), map_location=device)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    return state_dict


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


def _build_tapformer_predictor(eval_cfg, predictor_kwargs):
    device = _resolve_device(eval_cfg.get("device"))
    model = build_eval_model_from_config(eval_cfg, inference_mode="online")
    state_dict = _load_tapformer_checkpoint(eval_cfg, device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    predictor_kwargs = dict(predictor_kwargs)
    eval_resolution = _get_eval_resolution(eval_cfg)
    if eval_resolution is not None:
        predictor_kwargs["interp_shape"] = eval_resolution
    predictor = EvaluationPredictor(model, **predictor_kwargs)
    predictor.device = device
    return predictor


def _import_cowtracker_modules(repo_root):
    repo_root = str(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return {
        "CoWTracker": importlib.import_module("cowtracker.models.cowtracker").CoWTracker,
        "CoWTrackerOnline": importlib.import_module("cowtracker.models.cowtracker_online").CoWTrackerOnline,
        "CoWTrackerWindowed": importlib.import_module("cowtracker.models.cowtracker_windowed").CoWTrackerWindowed,
        "sample_dense_predictions": importlib.import_module("cowtracker.training.losses").sample_dense_predictions,
        "compute_padding_params": importlib.import_module("cowtracker.utils.padding").compute_padding_params,
        "apply_padding": importlib.import_module("cowtracker.utils.padding").apply_padding,
        "remove_padding_and_scale_back": importlib.import_module("cowtracker.utils.padding").remove_padding_and_scale_back,
    }


def _build_cowtracker_model(ckpt, cowtracker_cfg, bridge_cfg, repo_root, device, modules):
    if _model_uses_vggt(cowtracker_cfg):
        _ensure_vggt_available(repo_root)

    CoWTracker = modules["CoWTracker"]
    CoWTrackerOnline = modules["CoWTrackerOnline"]
    CoWTrackerWindowed = modules["CoWTrackerWindowed"]
    model_cfg_source = ckpt.get("config", cowtracker_cfg) if isinstance(ckpt, dict) else cowtracker_cfg
    model_kwargs = _get_cowtracker_model_kwargs(model_cfg_source)

    use_online = bool(bridge_cfg.get("online", True))
    data_cfg = cowtracker_cfg.get("data", {})
    online_backend = str(data_cfg.get("online_backend", "custom_online")).lower().strip()
    window_len = int(bridge_cfg.get("window_len") or data_cfg.get("seq_len", 8))
    window_stride = bridge_cfg.get("window_stride")
    if window_stride is None:
        window_stride = max(1, window_len // 2)
    else:
        window_stride = int(window_stride)
    num_memory_frames = bridge_cfg.get("num_memory_frames")
    if num_memory_frames is None:
        num_memory_frames = int(data_cfg.get("online_num_memory_frames", 10))
    else:
        num_memory_frames = int(num_memory_frames)

    if use_online:
        if online_backend == "official_windowed":
            model = CoWTrackerWindowed(
                window_len=window_len,
                stride=window_stride,
                num_memory_frames=num_memory_frames,
                **model_kwargs,
            ).to(device)
            model_call_mode = "official_windowed"
        elif online_backend == "custom_online":
            use_history_frames = bridge_cfg.get("use_history_frames")
            if use_history_frames is None:
                use_history_frames = bool(data_cfg.get("online_use_history_frames", True))
            init_mode = bridge_cfg.get("init_mode")
            if init_mode is None:
                init_mode = str(data_cfg.get("online_init_mode", "official")).lower().strip()
            model = CoWTrackerOnline(
                window_len=window_len,
                window_stride=window_stride,
                num_memory_frames=num_memory_frames,
                use_history_frames=bool(use_history_frames),
                init_mode=str(init_mode).lower().strip(),
                **model_kwargs,
            ).to(device)
            model_call_mode = "standard"
        else:
            raise ValueError("CowTracker data.online_backend must be one of: custom_online, official_windowed")
    else:
        model = CoWTracker(**model_kwargs).to(device)
        model_call_mode = "standard"

    state_dict = _normalize_state_dict_keys(_extract_state_dict(ckpt))
    legacy_prefixes = [
        "tracking_head.feature_extractor.",
        "tracking_head.aggregator.",
        "tracking_head.fnet.",
    ]
    if any(k.startswith(prefix) for k in state_dict for prefix in legacy_prefixes):
        state_dict = CoWTracker._remap_legacy_state_dict(state_dict)

    if use_online and online_backend == "official_windowed":
        if state_dict and not any(k.startswith("model.") for k in state_dict):
            state_dict = {f"model.{k}": v for k, v in state_dict.items()}
    elif state_dict and all(k.startswith("model.") for k in state_dict):
        state_dict = {k[len("model.") :]: v for k, v in state_dict.items()}

    incompatible = model.load_state_dict(state_dict, strict=False)
    print(
        "Loaded CowTracker checkpoint "
        f"(missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)})",
        flush=True,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, model_call_mode


def _build_cowtracker_predictor(eval_cfg):
    eval_model_cfg = _get_eval_model_config(eval_cfg)
    external_config = eval_model_cfg.get("external_config")
    if not external_config:
        raise ValueError("eval_model.external_config is required when eval_model.backend=cowtracker.")

    main_base_dir = _get_config_base_dir(eval_cfg)
    bridge_path = _resolve_path(external_config, main_base_dir)
    bridge_cfg = _load_yaml_config(bridge_path)
    bridge_backend = _normalize_eval_backend(bridge_cfg.get("backend", "cowtracker"))
    if bridge_backend != "cowtracker":
        raise ValueError(f"Unsupported external backend in {bridge_path}: {bridge_backend}")

    bridge_base_dir = bridge_path.parent
    repo_root = _resolve_path(bridge_cfg["repo_root"], bridge_base_dir)
    checkpoint_path = _resolve_path(bridge_cfg["checkpoint"], bridge_base_dir)
    model_config_path = _resolve_path(bridge_cfg["model_config"], bridge_base_dir)

    cowtracker_cfg = _load_yaml_config(model_config_path)
    modules = _import_cowtracker_modules(repo_root)
    device = _resolve_device(bridge_cfg.get("device"))
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    model, model_call_mode = _build_cowtracker_model(
        ckpt,
        cowtracker_cfg,
        bridge_cfg,
        repo_root,
        device,
        modules,
    )
    eval_resolution = _get_eval_resolution(eval_cfg)
    if eval_resolution is not None:
        infer_shape = eval_resolution
    else:
        infer_size_cfg = bridge_cfg.get("infer_size", "auto")
        infer_shape = None if str(infer_size_cfg).lower().strip() == "auto" else tuple(infer_size_cfg)

    predictor = CowTrackerEvaluationPredictor(
        model=model,
        sample_dense_predictions=modules["sample_dense_predictions"],
        compute_padding_params=modules["compute_padding_params"],
        apply_padding=modules["apply_padding"],
        remove_padding_and_scale_back=modules["remove_padding_and_scale_back"],
        infer_shape=infer_shape,
        use_padding=bool(bridge_cfg.get("use_padding", True)),
        skip_upscaling=bool(bridge_cfg.get("skip_upscaling", False)),
        amp_dtype=_autocast_dtype(bridge_cfg.get("precision", "bf16")),
        model_call_mode=model_call_mode,
        device=device,
    )
    return predictor


def build_eval_predictor_from_config(eval_cfg, **predictor_kwargs):
    backend = _get_eval_model_config(eval_cfg)["backend"]
    if backend == "tapformer_family":
        return _build_tapformer_predictor(eval_cfg, predictor_kwargs)
    if backend == "cowtracker":
        return _build_cowtracker_predictor(eval_cfg)
    raise ValueError(f"Unsupported eval backend: {backend}")
