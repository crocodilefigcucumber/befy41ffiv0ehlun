import torch
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F

import multiprocessing as mp
import os
import json
import pickle
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

# =========================
# Main Function
# =========================

REALIGNMENT_PATH = "trained_models/CUB"
PRECOMPUTED_PATH = "data/cub/output/cub_prediction_matrices.npz"
RESULTS_CSV = "results/CUB/test.csv"

if not os.path.exists(RESULTS_CSV):
    with open(RESULTS_CSV, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model_type", "test_loss", "test_acc"])

assert os.path.exists(PRECOMPUTED_PATH), (
    f"Error: Required file '{PRECOMPUTED_PATH}' not found. "
    "Please run the precomputation step first."
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_best_model_id(results_csv):
    results = pd.read_csv(results_csv)
    min_row = results[results["val_loss"]==results["val_loss"].min()]
    return min_row["run_idx"].iloc[0]

if __name__ == "__main__":
    # =========================
    # Load CBM
    # =========================
    cub_model_path = "models/cub_model_20250112_210439.pth"
    num_concepts = 312
    num_classes = 200

    # Set small_decoder to true to load the older model (no date in path.)
    m = model.ConceptBottleneckModel(num_concepts, num_classes, small_decoder=False)
    # Load the state dict
    state_dict = torch.load(cub_model_path, map_location=torch.device(device))
    m.load_state_dict(state_dict=state_dict)
    m.eval()

    # extract class predictor
    class_predictor = m.class_predictor

    # =========================
    # Load Test Data
    # =========================
    num_cpus = mp.cpu_count()
    num_workers = num_cpus - 2
    print(f"Number of CPUs: {num_cpus}")
    print(f"Number of workers: {num_workers}")

    data = np.load(PRECOMPUTED_PATH)
    precomputed_concepts = data["first"]

    # data_dict = cub.get_data_dict()
    # print("Creating Datasets")
    # _, _, test_dataset = cub.get_train_val_test_datasets(data_dict)
    # print("aslkdjasdkjalskd")

    file_path = "data.pkl"
    with open(file_path, "rb") as file:
        test_labels = pickle.load(file)

    test_data = list(zip(precomputed_concepts, test_labels))

    test_loader = DataLoader(
        test_data,
        batch_size=128,
        shuffle=False,  # No need to shuffle validation data
        num_workers=num_workers,
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
        val_results = f"{REALIGNMENT_PATH}/{network}/results.csv"
        if network != "Baseline":
            run_idx = get_best_model_id(val_results)
        
        try:
            with open(f"{REALIGNMENT_PATH}/{network}/run_{run_idx}_config.json", "r") as file:
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

        # load state dict
        state_dict = torch.load(
            f"{REALIGNMENT_PATH}/{network}/run_{run_idx}_best_model.pth", map_location=device
        )
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
            for concepts, labels in test_loader:
                concepts = concepts.to(device)
                labels = labels.to(device)
                labels = labels.squeeze()

                # ->RealignmentNetwork->...
                realigned_concepts = realign_concepts(
                    concept_corrector, concepts, device, config
                )
                # ->ClassPredictor
                predicted_labels = class_predictor(realigned_concepts)
                _, predicted = torch.max(predicted_labels.data, 1)

                test_total += labels.size(0)
                test_acc += (predicted == labels).sum().item()
                # labels_one_hot = F.one_hot(labels.squeeze(), num_classes=num_classes).float()
                # test_loss += criterion(predicted_labels,labels_one_hot)

        test_acc = 100 * test_acc / test_total
        test_loss = test_loss / test_total

        print(f"test_loss:{test_loss}, test_acc:{test_acc}")
        with open(RESULTS_CSV, mode="a", newline="") as file:  # open in append mode
            writer = csv.writer(file)
            writer.writerow([model_type, test_loss, test_acc])
