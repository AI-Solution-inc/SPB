import torchvision.models as models
import torch.nn as nn


def get_model_fcn_resnet50(pretrained=True, n_classes=8):
    '''
    # Аналогичный вариант, но с весами
    weights = FCN_ResNet50_Weights.DEFAULT
    transforms = weights.transforms(resize_size=None)

    model = fcn_resnet50(weights=weights, progress=False)
    '''
    model = models.segmentation.fcn_resnet50(pretrained=pretrained, progress=True)

    # change the classification FCNHead and make it learnable
    model.classifier[4] = nn.Conv2d(512, n_classes, kernel_size=(1, 1))

    # change the aux_classification FCNHead and make it learnable
    model.aux_classifier[4] = nn.Conv2d(256, n_classes, kernel_size=(1, 1))

    nn.init.xavier_uniform_(model.classifier[4].weight)
    nn.init.xavier_uniform_(model.aux_classifier[4].weight)

    return model


def get_model_fcn_resnet101(pretrained=True, n_classes=8):
    model = models.segmentation.fcn_resnet101(pretrained=pretrained, progress=True)

    # change the classification FCNHead and make it learnable
    model.classifier[4] = nn.Conv2d(512, n_classes, kernel_size=(1, 1))

    # change the aux_classification FCNHead and make it learnable
    model.aux_classifier[4] = nn.Conv2d(256, n_classes, kernel_size=(1, 1))

    nn.init.xavier_uniform_(model.classifier[4].weight)
    nn.init.xavier_uniform_(model.aux_classifier[4].weight)

    return model


def get_model_deeplabv3_resnet50(n_classes=8):
    weights = models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
    model = models.segmentation.deeplabv3_resnet50(weights)
    
    # change the classification FCNHead and make it learnable
    model.classifier[4] = nn.Conv2d(256, n_classes, kernel_size=(1, 1))

    # change the aux_classification FCNHead and make it learnable
    model.aux_classifier[4] = nn.Conv2d(256, n_classes, kernel_size=(1, 1))
    
    nn.init.xavier_uniform_(model.classifier[4].weight)
    nn.init.xavier_uniform_(model.aux_classifier[4].weight)

    return model


def get_model_deeplabv3_resnet101(n_classes=8):
    weights = models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT
    model = models.segmentation.deeplabv3_resnet101(weights)
    
    # change the classification FCNHead and make it learnable
    model.classifier[4] = nn.Conv2d(256, n_classes, kernel_size=(1, 1))

    # change the aux_classification FCNHead and make it learnable
    model.aux_classifier[4] = nn.Conv2d(256, n_classes, kernel_size=(1, 1))
    
    nn.init.xavier_uniform_(model.classifier[4].weight)
    nn.init.xavier_uniform_(model.aux_classifier[4].weight)

    return model
