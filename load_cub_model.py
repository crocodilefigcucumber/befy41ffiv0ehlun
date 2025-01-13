from models import model
import torch

import numpy as np

from data import cub

import evaluate


if __name__ == "__main__":
    print("Hello you bird lover you :D")
    cub_model_path = "models/cub_model_20250112_210439.pth"
    num_concepts = 312
    num_classes = 200
    # Set small_decoder to true to load the older model (no date in path.
    m = model.ConceptBottleneckModel(num_concepts, num_classes, small_decoder=False)
    # Load the state dict
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state_dict = torch.load(cub_model_path, map_location=torch.device(device))
    print(type(state_dict))
    m.load_state_dict(state_dict=state_dict)
    print(m)
    
    
    data_dict = cub.get_data_dict()
    print("Creating Datasets")
    train_dataset, val_dataset, test_dataset = cub.get_train_val_test_datasets(data_dict)
    print("Creating Dataloaders")
    train_loader, val_loader, test_loader = cub.get_train_val_test_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=512)

    train_acc, train_concept_acc, train_label_predictions, train_concept_predictions, _, _ = evaluate.evaluate_model(m, train_loader, "train", device)

    val_acc, val_concept_acc, val_label_predictions, val_concept_predictions, _, _ = evaluate.evaluate_model(m, val_loader, "val", device)
    
    test_acc, test_concept_acc, test_label_predictions, test_concept_predictions, test_label_goldens, test_concept_goldens = evaluate.evaluate_model(
        m, test_loader, "test", device)
    
    concept_prediction_mat_test = np.vstack(test_concept_predictions)
    label_predictions_mat_test = np.vstack(test_label_predictions)
    
    concept_prediction_mat_train = np.vstack(train_concept_predictions)
    label_predictions_mat_train = np.vstack(train_label_predictions)
    concept_prediction_mat_val = np.vstack(val_concept_predictions)
    label_predictions_mat_val = np.vstack(val_label_predictions) 
    
    test_concept_goldens_mat = np.vstack(test_concept_goldens)
    test_label_goldens_mat = np.vstack(test_label_goldens) 
    # Note the slightly weird order here.
    np.savez('data/cub/output/cub_prediction_matrices.npz',
             first = concept_prediction_mat_test,
             second = concept_prediction_mat_train,
             third = concept_prediction_mat_val,
             fourth = label_predictions_mat_test,
             fifth = label_predictions_mat_train,
             sixth = label_predictions_mat_val,
             seventh = test_concept_goldens_mat,
             eighth = test_label_goldens_mat)