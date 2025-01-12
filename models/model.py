import torch
import torch.nn as nn

from torchvision import models

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

# Define the concept bottleneck model
class ConceptBottleneckModel(nn.Module):
    def __init__(self, num_concepts=10, num_classes=200, small_decoder=False):
        super(ConceptBottleneckModel, self).__init__()
        self.encoder_res = models.resnet18(weights=None)
        self.encoder_res.load_state_dict(
            torch.load("./models/resnet18-5c106cde.pth")
        )
        #n_features = self.encoder_res.fc.in_features
        self.encoder_res.fc = Identity()
        self.features = nn.Sequential(self.encoder_res)

        # Concept predictor
        self.concept_predictor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, 1024), # this was (512, 256)
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, num_concepts),
                nn.Sigmoid()
            )


        # Class predictor
        if small_decoder:
            self.class_predictor = nn.Sequential(
                nn.Linear(num_concepts, 2048),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(2048, num_classes)
            )
        else:
            self.class_predictor = nn.Sequential(
                nn.Linear(num_concepts, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(2048, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, num_classes)
            )


    def forward(self, x, return_concepts=False):
        features = self.features(x)
        concepts = self.concept_predictor(features)
        outputs = self.class_predictor(concepts)

        if return_concepts:
            return outputs, concepts
        return outputs