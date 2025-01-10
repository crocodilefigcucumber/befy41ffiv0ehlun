import torch
from torch import nn

# =========================
# Baseline UCP Corrector Model
# =========================
class BaselineConceptCorrector(nn.Module):
    def __init__(self, input_size: int, output_size: int, input_format: str='original_and_intervened_inplace'):
        super(BaselineConceptCorrector, self).__init__()
        self.input_size = input_size
        self.input_format = input_format

    def forward(self, inputs: torch.Tensor, already_intervened_concepts: torch.Tensor, original_predictions: torch.Tensor):
        # No realignment, only replace with Groundtruth
        output = already_intervened_concepts * inputs + (1 - already_intervened_concepts) * original_predictions
        return output

    def forward_single_timestep(self, inputs: torch.Tensor, already_intervened_concepts: torch.Tensor, original_predictions: torch.Tensor):
        output = self.forward(inputs, already_intervened_concepts, original_predictions)
        return output

# =========================
# LSTM Concept Corrector Model
# =========================
class LSTMConceptCorrector(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, input_format: str='original_and_intervened_inplace'):
        super(LSTMConceptCorrector, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_format = input_format
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def prepare_initial_hidden(self, batch_size: int, device: torch.device):
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        )

    def forward(self, inputs: torch.Tensor, already_intervened_concepts: torch.Tensor, original_predictions: torch.Tensor, hidden):
        if self.input_format == 'original_and_intervened_inplace':
            x = already_intervened_concepts * inputs + (1 - already_intervened_concepts) * original_predictions
        elif self.input_format == 'previous_output':
            x = inputs
        else:
            raise ValueError(f"Unsupported input format: {self.input_format}")
        lstm_out, hid = self.lstm(x, hidden)
        output = torch.sigmoid(self.fc(lstm_out))
        output = already_intervened_concepts * inputs + (1 - already_intervened_concepts) * output
        return output, hid

    def forward_single_timestep(self, inputs: torch.Tensor, already_intervened_concepts: torch.Tensor, original_predictions: torch.Tensor, hidden, selected_clusters=None, selected_cluster_ids=None):
        inputs = inputs.unsqueeze(1)
        already_intervened_concepts = already_intervened_concepts.unsqueeze(1)
        original_predictions = original_predictions.unsqueeze(1)
        output, hid = self.forward(inputs, already_intervened_concepts, original_predictions, hidden)
        output = output.squeeze(1)
        return output, hid

# =========================
# Multi-Cluster LSTM Concept Corrector Model
# =========================
class MultiLSTMConceptCorrector(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, m_clusters: int, concept_to_cluster: list, input_format: str='original_and_intervened_inplace'):
        super(MultiLSTMConceptCorrector, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.m_clusters = m_clusters
        self.concept_to_cluster = concept_to_cluster
        self.input_format = input_format
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=len([c for c in range(input_size) if self.concept_to_cluster[c] == cluster_id]),
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
            )
            for cluster_id in range(m_clusters)
        ])
        self.fc_layers = nn.ModuleList([
            nn.Linear(
                hidden_size,
                len([c for c in range(input_size) if self.concept_to_cluster[c] == cluster_id])
            )
            for cluster_id in range(m_clusters)
        ])

    def prepare_initial_hidden(self, batch_size: int, device: torch.device):
        hidden_states = []
        for _ in range(self.m_clusters):
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            hidden_states.append((h_0, c_0))
        return hidden_states

    def forward(self, inputs: torch.Tensor, already_intervened_concepts: torch.Tensor, original_predictions: torch.Tensor, hidden_states: list):
        if self.input_format == 'original_and_intervened_inplace':
            x = already_intervened_concepts * inputs + (1 - already_intervened_concepts) * original_predictions
        elif self.input_format == 'previous_output':
            x = inputs
        else:
            raise ValueError(f"Unsupported input format: {self.input_format}")
        output = torch.zeros_like(x)
        for cluster_id in range(self.m_clusters):
            cluster_concept_indices = [c for c in range(self.input_size) if self.concept_to_cluster[c] == cluster_id]
            if not cluster_concept_indices:
                continue
            cluster_concepts = x[:, :, cluster_concept_indices]
            lstm_out, hid = self.lstm_layers[cluster_id](cluster_concepts, hidden_states[cluster_id])
            fc_out = torch.sigmoid(self.fc_layers[cluster_id](lstm_out))
            fc_out = already_intervened_concepts[:, :, cluster_concept_indices] * original_predictions[:, :, cluster_concept_indices] + \
                     (1 - already_intervened_concepts[:, :, cluster_concept_indices]) * fc_out
            output[:, :, cluster_concept_indices] = fc_out
            hidden_states[cluster_id] = hid
        return output, hidden_states

    def forward_single_timestep(self, inputs: torch.Tensor, already_intervened_concepts: torch.Tensor, original_predictions: torch.Tensor, hidden_states: list, selected_clusters=None, selected_cluster_ids=None):
        inputs = inputs.unsqueeze(1)
        already_intervened_concepts = already_intervened_concepts.unsqueeze(1)
        original_predictions = original_predictions.unsqueeze(1)
        output, updated_hidden_states = self.forward(
            inputs, already_intervened_concepts, original_predictions, hidden_states
        )
        output = output.squeeze(1)
        return output, updated_hidden_states