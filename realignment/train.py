# =========================
# Trajectory Sampling
# =========================
def sample_trajectory(concept_corrector: nn.Module, concepts: torch.Tensor, groundtruth_concepts: torch.Tensor, initial_hidden, intervention_policy, max_interventions: int, concept_to_cluster: list, model_type: str):
    with torch.no_grad():
        all_inputs = []
        all_masks = []
        all_original_predictions = []
        all_groundtruths = []
        if model_type == 'MultiLSTM':
            hidden_states = initial_hidden
        elif model_type == 'LSTM':
            hidden = initial_hidden
        elif model_type == 'Baseline':
            pass
        already_intervened_concepts = torch.zeros_like(concepts)
        original_predictions = concepts.detach().clone()
        all_inputs.append(concepts.detach().clone())
        all_masks.append(already_intervened_concepts.detach().clone())
        all_original_predictions.append(original_predictions)
        all_groundtruths.append(groundtruth_concepts.detach().clone())
        if model_type == 'MultiLSTM':
            out, hidden_states = concept_corrector.forward_single_timestep(
                concepts, already_intervened_concepts, original_predictions, hidden_states
            )
        elif model_type == 'LSTM':
            out, hidden = concept_corrector.forward_single_timestep(
                concepts, already_intervened_concepts, original_predictions, hidden
            )
        elif model_type == "Baseline":
            out = concept_corrector.forward_single_timestep(
                concepts, already_intervened_concepts, original_predictions
            )
        concepts = out
        for _ in range(max_interventions):
            concepts, already_intervened_concepts = intervene(
                concepts, already_intervened_concepts, groundtruth_concepts, intervention_policy
            )
            all_inputs.append(concepts.detach().clone())
            all_masks.append(already_intervened_concepts.detach().clone())
            all_original_predictions.append(original_predictions)
            all_groundtruths.append(groundtruth_concepts.detach().clone())
            if model_type == 'MultiLSTM':
                intervened = (concepts != original_predictions).float()
                concepts_to_intervene = torch.argmax(intervened, dim=1)
                selected_clusters = torch.tensor([concept_to_cluster[c.item()] for c in concepts_to_intervene]).to(concepts.device)
                unique_clusters = torch.unique(selected_clusters)
                out, hidden_states = concept_corrector.forward_single_timestep(
                    concepts, already_intervened_concepts, original_predictions, hidden_states, selected_clusters, unique_clusters.tolist()
                )
                concepts = out
            elif model_type == 'LSTM':
                out, hidden = concept_corrector.forward_single_timestep(
                    concepts, already_intervened_concepts, original_predictions, hidden
                )
                concepts = out
            elif model_type == 'Baseline':
                out = concept_corrector.forward_single_timestep(
                    concepts, already_intervened_concepts, original_predictions
                )
        all_inputs = torch.stack(all_inputs, dim=1)
        all_masks = torch.stack(all_masks, dim=1)
        all_original_predictions = torch.stack(all_original_predictions, dim=1)
        all_groundtruths = torch.stack(all_groundtruths, dim=1)
        if torch.min(all_inputs) < 0 or torch.max(all_inputs) > 1:
            print("Warning: All inputs have values outside the [0, 1] range.")
        return all_inputs, all_masks, all_original_predictions, all_groundtruths

# =========================
# Loss Computation
# =========================
def compute_loss(concept_corrector: nn.Module, concepts: torch.Tensor, groundtruth_concepts: torch.Tensor, initial_hidden, intervention_policy, max_interventions: int, criterion, concept_to_cluster: list, model_type: str):
    all_inputs, all_masks, all_original_predictions, all_groundtruths = sample_trajectory(
        concept_corrector, concepts, groundtruth_concepts, initial_hidden, intervention_policy, max_interventions, concept_to_cluster, model_type
    )
    if model_type == 'MultiLSTM':
        hidden_states = concept_corrector.prepare_initial_hidden(all_inputs.size(0), all_inputs.device)
        out, _ = concept_corrector.forward(all_inputs, all_masks, all_original_predictions, hidden_states)
    elif model_type == 'LSTM':
        out, _ = concept_corrector.forward(all_inputs, all_masks, all_original_predictions, None)
    elif model_type == 'Baseline':
        out = concept_corrector.forward(all_inputs,all_masks, all_original_predictions)
    return criterion(out, all_groundtruths)

# =========================
# Data Preparation and Splitting
# =========================
def ensure_all_classes(train_labels: torch.Tensor, J: int) -> bool:
    unique_classes = torch.unique(train_labels)
    return len(unique_classes) == J

def prepare_datasets(config: dict):
    data_gen_params = config['data_generation']
    k = data_gen_params['k']
    n_total = data_gen_params['n_total']
    n_train = data_gen_params['n_train']
    n_val = data_gen_params['n_val']
    J = data_gen_params['J']
    m = data_gen_params['m']
    seed = data_gen_params['seed']
    for attempt in range(1, 6):
        all_pred, all_gt, all_clusters, all_labels = generate_synthetic_data(k, n_total, J, m, seed + attempt)
        train_pred = all_pred[:n_train]
        train_gt = all_gt[:n_train]
        train_labels = all_labels[:n_train]
        val_pred = all_pred[n_train:n_train + n_val]
        val_gt = all_gt[n_train:n_train + n_val]
        val_labels = all_labels[n_train:n_train + n_val]
        if ensure_all_classes(train_labels, J):
            print(f"Data generation successful on attempt {attempt}.")
            break
        else:
            print(f"Attempt {attempt}: Not all classes present in training data. Regenerating...")
    else:
        raise ValueError("Failed to generate training data with all classes present after multiple attempts.")
    train_dataset = TensorDataset(
        train_pred.float(),
        train_gt.float()
    )
    val_dataset = TensorDataset(
        val_pred.float(),
        val_gt.float()
    )
    concept_to_cluster = [0] * k
    for cluster_id, concept_indices in all_clusters.items():
        for c in concept_indices:
            concept_to_cluster[c] = cluster_id
    return train_dataset, val_dataset, concept_to_cluster

# =========================
# Prepare DataLoaders
# =========================
def prepare_dataloaders(train_dataset: TensorDataset, val_dataset: TensorDataset, batch_size: int):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# =========================
# Main Training Loop
# =========================
def main():
    device = config['device']
    print(f"Using device: {device}")
    train_dataset, val_dataset, concept_to_cluster = prepare_datasets(config)
    train_loader, val_loader = prepare_dataloaders(
        train_dataset, val_dataset, config['batch_size']
    )
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    model_type = config['model']
    print(f"Selected model type: {model_type}")
    if model_type == 'MultiLSTM':
        ConceptCorrectorClass = MultiLSTMConceptCorrector
        concept_corrector = ConceptCorrectorClass(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            output_size=config['output_size'],
            m_clusters=config['data_generation']['m'],
            concept_to_cluster=concept_to_cluster,
            input_format=config['input_format']
        ).to(device)
    elif model_type == 'LSTM':
        ConceptCorrectorClass = LSTMConceptCorrector
        concept_corrector = ConceptCorrectorClass(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            output_size=config['output_size'],
            input_format=config['input_format']
        ).to(device)
    elif model_type == 'Baseline':
        ConceptCorrectorClass = BaselineConceptCorrector
        concept_corrector = ConceptCorrectorClass(
            input_size=config['input_size'],
            output_size=config['output_size'],
            input_format=config['input_format']
        ).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    print(f"{model_type} model initialized.")
    for name, param in concept_corrector.named_parameters():
        if 'weight' in name and param.ndimension() >= 2:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)
    print("Model weights initialized.")
    adapter = None
    if config['adapter_path'] is not None:
        if os.path.exists(config['adapter_path']):
            adapter = torch.load(config['adapter_path']).to(device)
            print(f"Using adapter from: {config['adapter_path']}")
        else:
            raise FileNotFoundError(f"Adapter path {config['adapter_path']} does not exist.")
    else:
        print("No adapter path provided. Skipping adapter loading.")
    # =========================
    # No Training for Baseline
    # =========================
    if model_type != 'Baseline':
        optimizer = optim.Adam(
            concept_corrector.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        print("Optimizer and scheduler initialized.")
    else:
        optimizer = None
        scheduler = None
        print("Skipping optimizer and scheduler for Baseline model")
    criterion = nn.BCELoss()
    print("Loss function initialized.")
    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = config['early_stop_patience']
    train_losses = []
    val_losses = []
    trained_models_dir = os.path.join('trained_models')
    os.makedirs(trained_models_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_filename = f"best_model_{model_type}_{timestamp}.pth"
    final_model_path = os.path.join(trained_models_dir, final_model_filename)
    for epoch in range(1, config['epochs'] + 1):
        start_time = time.time()
        concept_corrector.train()
        train_loss = 0.0
        if model_type != 'Baseline':
            for batch in train_loader:
                predicted_concepts, groundtruth_concepts = [b.to(device) for b in batch]
                optimizer.zero_grad()
                if model_type == 'MultiLSTM':
                    initial_hidden = concept_corrector.prepare_initial_hidden(predicted_concepts.size(0), device)
                else:
                    initial_hidden = concept_corrector.prepare_initial_hidden(predicted_concepts.size(0), device)
                if adapter is not None:
                    predicted_concepts, _ = adapter.forward_single_timestep(
                        predicted_concepts, torch.zeros_like(predicted_concepts), predicted_concepts, initial_hidden
                    )
                loss = compute_loss(
                    concept_corrector,
                    predicted_concepts,
                    groundtruth_concepts,
                    initial_hidden,
                    ucp,
                    config['max_interventions'],
                    criterion,
                    concept_to_cluster,
                    model_type
                )
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            scheduler.step()
            average_train_loss = train_loss / len(train_loader)
            train_losses.append(average_train_loss)
            concept_corrector.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    predicted_concepts, groundtruth_concepts = [b.to(device) for b in batch]
                    if model_type == 'MultiLSTM':
                        initial_hidden = concept_corrector.prepare_initial_hidden(predicted_concepts.size(0), device)
                    elif model_type == 'LSTM':
                        initial_hidden = concept_corrector.prepare_initial_hidden(predicted_concepts.size(0), device)
                    elif model_type == 'Baseline':
                        pass
                    if adapter is not None:
                        predicted_concepts, _ = adapter.forward_single_timestep(
                            predicted_concepts, torch.zeros_like(predicted_concepts), predicted_concepts, initial_hidden
                        )
                    loss = compute_loss(
                        concept_corrector,
                        predicted_concepts,
                        groundtruth_concepts,
                        initial_hidden,
                        ucp,
                        config['max_interventions'],
                        criterion,
                        concept_to_cluster,
                        model_type
                    )
                    val_loss += loss.item()
            average_val_loss = val_loss / len(val_loader)
            val_losses.append(average_val_loss)
            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                early_stop_counter = 0
                print(f"Epoch {epoch}: Improved validation loss to {best_val_loss:.4f}.")
            else:
                early_stop_counter += 1
                print(f"Epoch {epoch}: Validation loss did not improve ({average_val_loss:.4f}).")
            epoch_time = time.time() - start_time
            print(f"Epoch [{epoch}/{config['epochs']}], "
                f"Train Loss: {average_train_loss:.4f}, "
                f"Val Loss: {average_val_loss:.4f}, "
                f"Time: {epoch_time:.2f}s")
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break
    torch.save(concept_corrector.state_dict(), final_model_path)
    print(f"Best model saved to {final_model_path}")
    print("Training completed.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    plt.figure(figsize=(10,5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Over Epochs')
    plt.show()

if __name__ == '__main__':
    main()