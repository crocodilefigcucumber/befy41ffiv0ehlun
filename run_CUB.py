import torch
import numpy as np

import os
import re

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

PRECOMPUTED_PATH = "data/cub/output/cub_prediction_matrices.npz"
REALIGNMENT_PATH = "trained_models/CUB"

if __name__ == "__main__":
    # =========================
    # Evaluate Concept predictor X->c
    # =========================

    # First check whether predictions already exist:
    if not os.path.isfile(PRECOMPUTED_PATH):
        print("No precomputed predictions found, computing them now ...")

        cub_model_path = "models/cub_model.pth"
        num_concepts = 312
        num_classes = 200
        m = model.ConceptBottleneckModel(num_concepts, num_classes)
        # Load the state dict
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        state_dict = torch.load(cub_model_path, map_location=torch.device(device))
        m.load_state_dict(state_dict=state_dict)

        data_dict = cub.get_data_dict()
        print("Creating Datasets")
        train_dataset, val_dataset, test_dataset = cub.get_train_val_test_datasets(
            data_dict
        )
        print("Creating Dataloaders")
        train_loader, val_loader, test_loader = cub.get_train_val_test_loaders(
            train_dataset, val_dataset, test_dataset, batch_size=512
        )

        (
            train_acc,
            train_concept_acc,
            train_label_predictions,
            train_concept_predictions,
        ) = evaluate.evaluate_model(m, train_loader, "train", device)

        val_acc, val_concept_acc, val_label_predictions, val_concept_predictions = (
            evaluate.evaluate_model(m, val_loader, "val", device)
        )

        test_acc, test_concept_acc, test_label_predictions, test_concept_predictions = (
            evaluate.evaluate_model(m, test_loader, "test", device)
        )

        concept_prediction_mat_test = np.vstack(test_concept_predictions)
        label_predictions_mat_test = np.vstack(test_label_predictions)
        concept_prediction_mat_train = np.vstack(train_concept_predictions)
        label_predictions_mat_train = np.vstack(train_label_predictions)
        concept_prediction_mat_val = np.vstack(val_concept_predictions)
        label_predictions_mat_val = np.vstack(val_label_predictions)

        np.savez(
            PRECOMPUTED_PATH,
            first=concept_prediction_mat_test,
            second=concept_prediction_mat_train,
            third=concept_prediction_mat_val,
            fourth=label_predictions_mat_test,
            fifth=label_predictions_mat_train,
            sixth=label_predictions_mat_val,
        )
        print("Predictions saved.")
    else:
        print("Precomputed predictions found.")

    # =========================
    # Load Realignment Networks
    # =========================

    NETWORKS = os.listdir(REALIGNMENT_PATH)
    print(NETWORKS)
    for realignment_network_filename in NETWORKS:
        # extract model type from its path
        model_type = re.search(
            r"best_model_(.*?)_", realignment_network_filename
        ).group(1)

        device = config["device"]
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

        
