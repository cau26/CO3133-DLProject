"""Object detection model builders for Assignment 2:
1. SSDLite320-MobileNetV3-Large (lightweight baseline, ImageNet backbone + 21-class heads)
2. Faster R-CNN ResNet-50-FPN (COCO_V1 pretrained detector, frozen backbone vs. full fine-tuning)
"""
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .common import set_seed


def build_detector(name, config):
    model_cfg = config["models"][name]
    arch = model_cfg["architecture"]
    num_classes = config["num_classes"]
    set_seed(config["seed"])

    if arch == "ssdlite320_mobilenet_v3_large":
        model = ssdlite320_mobilenet_v3_large(
            weights=None,
            weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1,
            num_classes=num_classes,
            score_thresh=config["score_threshold"],
            nms_thresh=config["nms_threshold"],
            detections_per_img=config["max_detections_per_image"],
        )
    elif arch == "fasterrcnn_resnet50_fpn":
        model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
            trainable_backbone_layers=5,
            min_size=model_cfg["min_size"],
            max_size=model_cfg["max_size"],
            box_score_thresh=config["score_threshold"],
            box_nms_thresh=config["nms_threshold"],
            box_detections_per_img=config["max_detections_per_image"],
        )
        # Re-seed before replacing the predictor so frozen and finetune runs share identical head init
        set_seed(config["seed"])
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        if model_cfg.get("freeze_backbone", False):
            # Freeze the ResNet-50 body while keeping FPN, RPN, and ROI heads trainable (Proposal Sec. 2.4)
            for param in model.backbone.body.parameters():
                param.requires_grad = False
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    return model


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}
