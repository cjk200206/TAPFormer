import torch.nn as nn

from LFE_TAP.models.fusionFormer import Fusionformer, Unet_Transformer
from LFE_TAP.models.tapformer import TAPFormer


class _SingleModalityFeatureAdapter(nn.Module):
    """Adapter with TAPFormer fusion_block-compatible signature."""

    def __init__(self, modality: str, image_size=(384, 512), out_dim=128, stride=8):
        super().__init__()
        mode = modality.lower()
        if mode not in {"image", "event"}:
            raise ValueError(f"Unsupported modality: {modality}")
        self.modality = mode
        input_dim = 3 if mode == "image" else 10
        self.backbone = Unet_Transformer(
            input_dim=input_dim,
            image_size=image_size,
            out_dim=out_dim,
            mlp_dim=512,
            depth=2,
            stride=stride,
            dropout=0.0,
        )

    def forward(self, x_i, x_e, img_ifnew=None, feature_teacher=None):
        if self.modality == "image":
            return self.backbone(x_i, feature_teacher=feature_teacher)
        return self.backbone(x_e, feature_teacher=feature_teacher)


class TAPFormerAblation(TAPFormer):
    """
    TAPFormer variant with configurable front-end for ablations.

    feature_mode:
      - "fusion": original image+event fusion
      - "image": image-only front-end
      - "event": event-only front-end
    """

    def __init__(
        self,
        window_size=16,
        stride=8,
        corr_radius=3,
        corr_levels=3,
        backbone="basic",
        num_heads=8,
        hidden_size=384,
        space_depth=3,
        time_depth=3,
        feature_mode="fusion",
    ):
        super().__init__(
            window_size=window_size,
            stride=stride,
            corr_radius=corr_radius,
            corr_levels=corr_levels,
            backbone=backbone,
            num_heads=num_heads,
            hidden_size=hidden_size,
            space_depth=space_depth,
            time_depth=time_depth,
        )

        self.feature_mode = str(feature_mode).lower().strip()
        if self.feature_mode in {"fusion", "fused"}:
            self.fusion_block = Fusionformer(
                image_size=self.model_resolution,
                out_dim=self.latent_dim,
                mlp_dim=512,
                stride=self.stride,
                depth=2,
            )
        elif self.feature_mode in {"image", "rgb"}:
            self.fusion_block = _SingleModalityFeatureAdapter(
                modality="image",
                image_size=self.model_resolution,
                out_dim=self.latent_dim,
                stride=self.stride,
            )
        elif self.feature_mode in {"event", "events"}:
            self.fusion_block = _SingleModalityFeatureAdapter(
                modality="event",
                image_size=self.model_resolution,
                out_dim=self.latent_dim,
                stride=self.stride,
            )
        else:
            raise ValueError(
                f"Unsupported feature_mode: {feature_mode}. "
                "Use one of: fusion, image, event."
            )
