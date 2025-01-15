import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

import os
import json
import csv

from models import model

from realignment.concept_corrector_models import (
    BaselineConceptCorrector,
    LSTMConceptCorrector,
    MultiLSTMConceptCorrector,
    GRUConceptCorrector,
    RNNConceptCorrector,
    MultiGRUConceptCorrector,
    MultiRNNConceptCorrector,
)

from realignment.data_loader import load_data, create_dataloaders
from realignment.realign_concepts import realign_concepts
from realignment.intervention_utils import ucp

# =========================
# Main Function
# =========================

REALIGNMENT_PATH = "trained_models/CUB"
PRECOMPUTED_PATH = "data/cub/output/cub_prediction_matrices.npz"
RESULTS_CSV = "results/CUB/test.csv"


with open(RESULTS_CSV, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["model_type", "test_acc"])

assert os.path.exists(PRECOMPUTED_PATH), (
    f"Error: Required file '{PRECOMPUTED_PATH}' not found. "
    "Please run the precomputation step first."
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_best_model_id(results_csv):
    results = pd.read_csv(results_csv)
    min_row = results[results["val_loss"] == results["val_loss"].min()]
    return min_row["run_idx"].iloc[0]


if __name__ == "__main__":
    # =========================
    # Load CBM
    # =========================
    cub_model_path = (
        "models/cub_model_20250113_212453.pth"  # "models/cub_model_20250112_210439.pth"
    )
    num_concepts = 312
    num_classes = 200

    # Set small_decoder to true to load the older model (no date in path.)
    m = model.ConceptBottleneckModel(num_concepts, num_classes, small_decoder=False)
    # Load the state dict
    state_dict = torch.load(cub_model_path, map_location=torch.device(device))
    m.load_state_dict(state_dict=state_dict)
    m.eval()
    m.to(device)

    # extract class predictor
    class_predictor = m.class_predictor

    # =========================
    # Load Test Data
    # =========================
    num_cpus = 5
    num_workers = num_cpus - 2
    print(f"Number of CPUs: {num_cpus}")
    print(f"Number of workers: {num_workers}")

    data = np.load(PRECOMPUTED_PATH)
    precomputed_concepts = data["first"]
    test_labels = data["eighth"]
    test_concepts = data["seventh"]

    test_data = list(zip(precomputed_concepts, test_concepts, test_labels))

    test_loader = DataLoader(
        test_data,
        batch_size=128,
        shuffle=False,  # No need to shuffle validation data
        num_workers=num_workers,
        pin_memory=True,
    )

    # =========================
    # Test Full Pipeline X->ConceptEncoder->RealignmentNetwork->ClassPredictor
    # =========================
    NETWORKS = os.listdir(REALIGNMENT_PATH)
    NETWORKS = [network for network in NETWORKS if "maxinter_" not in network]

    for network in NETWORKS:
        # =========================
        # Load Realignment Network
        # =========================

        print(f"Testing {network}")
        # gather config and model type
        val_results = f"{REALIGNMENT_PATH}/{network}/results.csv"
        run_idx = get_best_model_id(val_results)

        try:
            with open(
                f"{REALIGNMENT_PATH}/{network}/run_{run_idx}_config.json", "r"
            ) as file:
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
        elif model_type == "GRU":
            ConceptCorrectorClass = GRUConceptCorrector
            concept_corrector = ConceptCorrectorClass(
                input_size=input_size,
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"],
                output_size=output_size,
                input_format=config["input_format"],
            ).to(device)
        elif model_type == "RNN":
            ConceptCorrectorClass = RNNConceptCorrector
            concept_corrector = ConceptCorrectorClass(
                input_size=input_size,
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"],
                output_size=output_size,
                input_format=config["input_format"],
            ).to(device)
        elif model_type == "MultiGRU":
            ConceptCorrectorClass = MultiGRUConceptCorrector
            concept_corrector = ConceptCorrectorClass(
                input_size=input_size,
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"],
                output_size=output_size,
                m_clusters=number_clusters,
                concept_to_cluster=concept_to_cluster,
                input_format=config["input_format"],
            ).to(device)
        elif model_type == "MultiRNN":
            ConceptCorrectorClass = MultiRNNConceptCorrector
            concept_corrector = ConceptCorrectorClass(
                input_size=input_size,
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"],
                output_size=output_size,
                m_clusters=number_clusters,
                concept_to_cluster=concept_to_cluster,
                input_format=config["input_format"],
            ).to(device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        print(f"{model_type} model initialized.")

        # load state dict if not Baseline, Baseline doesn't need to load weights
        if network != "Baseline":
            state_dict = torch.load(
                f"{REALIGNMENT_PATH}/{network}/run_{run_idx}_best_model.pth",
                map_location=device,
            )
            print("Loading state dict.")
            concept_corrector.load_state_dict(state_dict)

        # =========================
        # Testing Loop
        # =========================

        test_total = 0
        test_acc = 0.0

        with torch.no_grad():
            for concepts, ground_truth, labels in test_loader:
                concepts = concepts.to(device)
                labels = labels.to(device)
                ground_truth = ground_truth.to(device)
                labels = labels.squeeze()

                # ->RealignmentNetwork->...
                realigned_concepts = realign_concepts(
                    concept_corrector=concept_corrector,
                    concept_vector=concepts,
                    groundtruth_concepts=ground_truth,
                    intervention_policy=ucp,
                    device=device,
                    config=config,
                    concept_to_cluster=None,
                    verbose=False,
                )
                # ->ClassPredictor
                predicted_labels = class_predictor(realigned_concepts)

                _, predicted = torch.max(predicted_labels.data, 1)

                # print(
                #     f"Change: {(concepts[0]-realigned_concepts[0]).abs()}, Predict:{predicted[0]}"
                # )

                test_total += labels.size(0)
                test_acc += (predicted == labels).sum().item()

        test_acc = 100 * test_acc / test_total

        print(f"test_acc:{test_acc}")
        with open(RESULTS_CSV, mode="a", newline="") as file:  # open in append mode
            writer = csv.writer(file)
            writer.writerow([model_type, test_acc])
