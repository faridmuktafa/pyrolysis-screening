from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
def build_model(num_classes, pretrained=True):
    model=maskrcnn_resnet50_fpn(weights='DEFAULT' if pretrained else None)
    in_feat=model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor=FastRCNNPredictor(in_feat, num_classes)
    in_feat_mask=model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor=MaskRCNNPredictor(in_feat_mask,256,num_classes)
    return model
