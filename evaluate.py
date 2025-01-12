import torch

# Evaluate model
def evaluate_model(model, test_dataloader, split="test", device='cuda'):
    model.eval()  # Set to evaluation mode
    val_acc = 0.0
    concept_acc = 0.0
    val_total = 0
    concept_total = 0
    concept_predictions = []
    label_predictions = []

    with torch.no_grad():
        for images, concepts, labels in test_dataloader:
            images = images.to(device)
            concepts = concepts.to(device)
            labels = labels.to(device)

            predicted_labels, predicted_concepts = model(images, return_concepts=True)
            _, predicted = torch.max(predicted_labels.data, 1)
            label_predictions.append(predicted_labels.numpy())
            val_total += labels.size(0)
            val_acc += (predicted == labels).sum().item()
            concept_total += concepts.size(0) * concepts.size(1)
            preds = (predicted_concepts > 0.5).float()  # Need to threshold sigmoid outputs
            concept_predictions.append(predicted_concepts.numpy())
            concept_acc += (preds == concepts).float().sum()

    val_acc = 100 * val_acc / val_total
    concept_acc = 100 * concept_acc / concept_total
    #print(f'Validation Loss: {running_loss/len(train_loader):.4f}')
    print(f'{split.capitalize()} Class Label Accuracy: {val_acc:.2f}%')
    print(f'{split.capitalize()} Concept Accuracy: {concept_acc:.2f}%')
    if split != "test":
        print(f"Warning, evaluating predictions on {split}, "
               "be careful if using for downstream applications.")
    return val_acc, concept_acc, label_predictions, concept_predictions