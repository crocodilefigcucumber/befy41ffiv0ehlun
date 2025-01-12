from models import model
import torch
import os

import numpy as np

from data import cub

import evaluate

# =========================
# Main Function
# =========================

PRECOMPUTED_PATH = 'data/cub/output/cub_prediction_matrices.npz'


if __name__ == "__main__":
    # =========================
    # Evaluate Concept predictor X->c
    # =========================

    # First check whether predictions already exist:
    if not os.isfile(PRECOMPUTED_PATH):
        print("No precomputed predictions found, computing them now ...")

        cub_model_path = "models/cub_model.pth"
        num_concepts = 312
        num_classes = 200
        m = model.ConceptBottleneckModel(num_concepts, num_classes)
        # Load the state dict
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        state_dict = torch.load(cub_model_path, map_location=torch.device(device))
        m.load_state_dict(state_dict=state_dict)
        
        
        data_dict = cub.get_data_dict()
        print("Creating Datasets")
        train_dataset, val_dataset, test_dataset = cub.get_train_val_test_datasets(data_dict)
        print("Creating Dataloaders")
        train_loader, val_loader, test_loader = cub.get_train_val_test_loaders(
            train_dataset, val_dataset, test_dataset, batch_size=512)

        train_acc, train_concept_acc, train_label_predictions, train_concept_predictions = evaluate.evaluate_model(m, train_loader, "train", device)

        val_acc, val_concept_acc, val_label_predictions, val_concept_predictions = evaluate.evaluate_model(m, val_loader, "val", device)
        
        test_acc, test_concept_acc, test_label_predictions, test_concept_predictions = evaluate.evaluate_model(m, test_loader, "test", device)
        
        concept_prediction_mat_test = np.vstack(test_concept_predictions)
        label_predictions_mat_test = np.vstack(test_label_predictions)
        concept_prediction_mat_train = np.vstack(train_concept_predictions)
        label_predictions_mat_train = np.vstack(train_label_predictions)
        concept_prediction_mat_val = np.vstack(val_concept_predictions)
        label_predictions_mat_val = np.vstack(val_label_predictions) 
        # Note the slightly weird order here.
        np.savez(PRECOMPUTED_PATH,
                first = concept_prediction_mat_test,
                second = concept_prediction_mat_train,
                third = concept_prediction_mat_val,
                fourth = label_predictions_mat_test,
                fifth = label_predictions_mat_train,
                sixth = label_predictions_mat_val)
        print("Predictions saved.")
    else:
        print("Precomputed predictions found.")
    
    # =========================
    # Load Realignment Networks
    # =========================

    