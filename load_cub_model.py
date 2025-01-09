from models import model
import torch

import numpy as np

from data import cub

import evaluate


if __name__ == "__main__":
    print("Hello you bird lover you :D")
    cub_model_path = "models/cub_model.pth"
    num_concepts = 312
    num_classes = 200
    m = model.ConceptBottleneckModel(num_concepts, num_classes)
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
    train_loader, val_loader, test_loader = cub.get_train_val_test_loaders(train_dataset, val_dataset, test_dataset, batch_size=512)

    val_acc, concept_acc, label_predictions, concept_predictions = evaluate.evaluate_model(m, test_loader, device)
    
    concept_prediction_mat = np.vstack(concept_predictions)
    label_predictions_mat = np.vstack(label_predictions)
    np.savez('data/cub/output/cub_prediction_matrices.npz', first=concept_prediction_mat, second=label_predictions_mat)
