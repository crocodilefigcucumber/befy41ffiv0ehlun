import torch
from torch import nn
import os
import json
import itertools
from config import config as default_config
from concept_corrector_models import (
    BaselineConceptCorrector,
    LSTMConceptCorrector,
    MultiLSTMConceptCorrector,
    GRUConceptCorrector,
    RNNConceptCorrector,
    MultiGRUConceptCorrector,
    MultiRNNConceptCorrector,
)
from train import train_model
from eval import evaluate_baseline
from data_loader import load_data, create_dataloaders, CustomDataset
import csv

GRID_SEARCH_PARAMS = {"hidden_size": [64, 128, 256], "hidden_layers": [1, 3, 5]}
grid = list(
    itertools.product(
        GRID_SEARCH_PARAMS["hidden_size"], GRID_SEARCH_PARAMS["hidden_layers"]
    )
)


def CV(config, grid):
    # Initialize CSV with headers if file doesn't exist
    results_csv = f"trained_models/{config['dataset']}/{config['model']}/results.csv"
    if not os.path.exists(results_csv):
        with open(results_csv, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["run_idx", "hidden_size", "hidden_layers", "val_loss"])

    for run_idx, (hidden_size, hidden_layers) in enumerate(grid):
        # Update configuration with current hyperparameters
        config["hidden_size"] = hidden_size
        config["num_layers"] = hidden_layers

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

        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            predicted_concepts, groundtruth_concepts, config
        )
        model_type = config["model"]
        print(f"Selected model type: {model_type}")

        # Initialize the appropriate Concept Corrector model
        if model_type == "MultiLSTM":
            concept_corrector = MultiLSTMConceptCorrector(
                input_size=input_size,
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"],
                output_size=output_size,
                m_clusters=number_clusters,
                concept_to_cluster=concept_to_cluster,
                input_format=config["input_format"],
            ).to(device)
        elif model_type == "LSTM":
            concept_corrector = LSTMConceptCorrector(
                input_size=input_size,
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"],
                output_size=output_size,
                input_format=config["input_format"],
            ).to(device)
        elif model_type == "Baseline":
            concept_corrector = BaselineConceptCorrector(
                input_size=input_size,
                output_size=output_size,
                input_format=config["input_format"],
            ).to(device)
        elif model_type == "GRU":
            concept_corrector = GRUConceptCorrector(
                input_size=input_size,
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"],
                output_size=output_size,
                input_format=config["input_format"],
            ).to(device)
        elif model_type == "RNN":
            concept_corrector = RNNConceptCorrector(
                input_size=input_size,
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"],
                output_size=output_size,
                input_format=config["input_format"],
            ).to(device)
        elif model_type == "MultiGRU":
            concept_corrector = MultiGRUConceptCorrector(
                input_size=input_size,
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"],
                output_size=output_size,
                m_clusters=number_clusters,
                concept_to_cluster=concept_to_cluster,
                input_format=config["input_format"],
            ).to(device)
        elif model_type == "MultiRNN":
            concept_corrector = MultiRNNConceptCorrector(
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

        # Initialize model weights
        for name, param in concept_corrector.named_parameters():
            if "weight" in name and param.ndimension() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)
        print("Model weights initialized.")

        # Load adapter if provided
        adapter = None
        if config["adapter_path"] is not None:
            if os.path.exists(config["adapter_path"]):
                adapter = torch.load(config["adapter_path"]).to(device)
                print(f"Using adapter from: {config['adapter_path']}")
            else:
                raise FileNotFoundError(
                    f"Adapter path {config['adapter_path']} does not exist."
                )
        else:
            print("No adapter path provided. Skipping adapter loading.")

        val_loss = None
        # Train the model if not Baseline
        if config["model"] != "Baseline":
            val_loss = train_model(
                concept_corrector,
                train_loader,
                val_loader,
                device,
                config,
                concept_to_cluster,
                adapter,
                run_idx,
            )
        else:
            baseline_dir = os.path.join("trained_models", config["dataset"], "Baseline")
            os.makedirs(baseline_dir, exist_ok=True)
            config_save_path = os.path.join(baseline_dir, "config.json")
            with open(config_save_path, "w") as f:
                json.dump(config, f, indent=4)
            print(f"Configuration saved to {config_save_path}")

            print("\nBaseline Model Evaluation with Interventions:")
            verbose = config["verbose"]
            print("Intervening on Training Data:")
            evaluate_baseline(
                concept_corrector,
                train_loader,
                device,
                config,
                concept_to_cluster,
                adapter,
                phase="Training",
                verbose=verbose,
            )
            print("Intervening on Validation Data:")
            val_loss = evaluate_baseline(
                concept_corrector,
                val_loader,
                device,
                config,
                concept_to_cluster,
                adapter,
                phase="Validation",
                verbose=verbose,
            )
            print("Baseline model evaluation completed.")

        # Append results to CSV
        with open(results_csv, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    run_idx,
                    config["hidden_size"],
                    config["num_layers"],
                    val_loss if val_loss is not None else "N/A",
                ]
            )
        print(f"Appended results to {results_csv} for run {run_idx}.")


# =========================
# Main Function
# =========================
def main():
    model_types = [
        "Baseline",
        "LSTM",
        "MultiLSTM",
        "GRU",
        "RNN",
        "MultiGRU",
        "MultiRNN",
    ]
    for model in model_types:
        config = default_config.copy()
        config["model"] = model
        CV(config, grid)


if __name__ == "__main__":
    main()
