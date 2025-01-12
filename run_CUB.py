import torch
from torch import nn
import numpy as np

import multiprocessing as mp
import os
import json

from models import model
from data import cub

from realignment.concept_corrector_models import (
    BaselineConceptCorrector,
    LSTMConceptCorrector,
    MultiLSTMConceptCorrector,
)

from realignment.data_loader import load_data, create_dataloaders

import evaluate

# =========================
# Main Function
# =========================

REALIGNMENT_PATH = "trained_models/CUB"

if __name__ == "__main__":
    # =========================
    # Load CBM
    # =========================
    cub_model_path = "models/cub_model.pth"
    num_concepts = 312
    num_classes = 200
    m = model.ConceptBottleneckModel(num_concepts, num_classes)
    # Load the state dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = torch.load(cub_model_path, map_location=torch.device(device))
    m.load_state_dict(state_dict=state_dict)
    m.eval()

    # extract class predictor
    class_predictor = m.class_predictor

    data_dict = cub.get_data_dict()
    print("Creating Datasets")
    train_dataset, val_dataset, test_dataset = cub.get_train_val_test_datasets(
        data_dict
    )
    print("Creating Dataloaders")
    train_loader, val_loader, test_loader = cub.get_train_val_test_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=512
    )

    # =========================
    # Test Full Pipeline X->ConceptEncoder->RealignmentNetwork->ClassPredictor
    # =========================
    NETWORKS = os.listdir(REALIGNMENT_PATH)
    for network in NETWORKS:
        # =========================
        # Load Realignment Network
        # =========================

        print(f"Testing {network}")
        # gather config and model type
        try:
            with open(f'{REALIGNMENT_PATH}/{network}/config.json', 'r') as file:
                config = json.load(file)
        except FileNotFoundError:
            print("The config file was not found.")
        except json.JSONDecodeError:
            print("Error decoding config JSON.")
        
        model_type = config["model"]

        print(f"Using device: {device}")
        (
            predicted_concepts,
            groundtruth_concepts,
            concept_to_cluster,
            input_size,
            output_size,
            number_clusters,
        ) = load_data(config)

        # Initialize the appropriate Concept Corrector model
        if model_type == "MultiLSTM":
            ConceptCorrectorClass = MultiLSTMConceptCorrector
            concept_corrector = ConceptCorrectorClass(
                input_size=input_size,
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"],
                output_size=output_size,
                m_clusters=number_clusters,
                concept_to_cluster=concept_to_cluster,
                input_format=config["input_format"],
            ).to(device)
        elif model_type == "LSTM":
            ConceptCorrectorClass = LSTMConceptCorrector
            concept_corrector = ConceptCorrectorClass(
                input_size=input_size,
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"],
                output_size=output_size,
                input_format=config["input_format"],
            ).to(device)
        elif model_type == "Baseline":
            ConceptCorrectorClass = BaselineConceptCorrector
            concept_corrector = ConceptCorrectorClass(
                input_size=input_size,
                output_size=output_size,
                input_format=config["input_format"],
            ).to(device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        print(f"{model_type} model initialized.")

        #load state dict
        state_dict = torch.load(f'{REALIGNMENT_PATH}/{network}/best_model.pth', map_location=device)
        concept_corrector.load_state_dict(state_dict)

        # =========================
        # Testing Loop
        # =========================

        # initialize loss function
        criterion = nn.BCELoss()
        print("Loss function initialized.")

        test_loss = 0.0
        test_total = 0
        test_acc = 0.0

        with torch.no_grad():
            for images, concepts, labels in test_loader:
                images = images.to(device)
                concepts = concepts.to(device)
                labels = labels.to(device)

                # X->ConceptEncoder
                _, predicted_concepts = model(images, return_concepts=True)
                # ->RealignmentNetwork->...
                realigned_concepts = concept_corrector(predicted_concepts)
                # ->ClassPredictor
                predicted_labels = class_predictor(realigned_concepts)

                _, predicted = torch.max(predicted_labels.data, 1)

                test_total += labels.size(0)
                test_acc += (predicted == labels).sum().item()
                test_loss += criterion(predicted_labels,labels)

        test_acc = 100 * test_acc / test_total
        test_loss = test_loss / test_total

        print(test_loss, test_acc)



        
                



        
