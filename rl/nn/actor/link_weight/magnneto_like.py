import torch
import torch.nn.functional as F
from torch import nn as nn, Tensor
from torch_geometric.data import Batch
from torch_scatter import scatter_max, scatter_min

from rl.nn.actor.link_weight._link_weight_actor import LinkWeightActor


class MagnnetoLikeActor(LinkWeightActor):
    """
    Link-weight actor network architecture as taken from the official MAGNNETO implementation
    (but translated to PyTorch).
    """

    # Values taken from the official MAGNNETO implementation
    NUM_ACTIONS = 1
    NUM_FEATURES = 2
    LINK_STATE_SIZE = 16
    FIRST_HIDDEN_LAYER_SIZE = 128
    DROPOUT_RATE = 0.5
    FINAL_HIDDEN_LAYER_SIZE = 64
    MESSAGE_ITERATIONS = 8

    def __init__(self, device: str):
        super(MagnnetoLikeActor, self).__init__(device)

        # INITIALIZERS
        self.hidden_layer_initializer = nn.init.orthogonal_
        self.final_layer_initializer = lambda x: nn.init.orthogonal_(x, gain=0.01)

        # NEURAL NETWORKS
        self.create_message = nn.Sequential(
            nn.Linear(2 * self.LINK_STATE_SIZE, self.FINAL_HIDDEN_LAYER_SIZE, device=device),
            nn.Tanh(),
            nn.Linear(self.FINAL_HIDDEN_LAYER_SIZE, self.LINK_STATE_SIZE, device=device),
            nn.Tanh(),
        )

        self.link_update = nn.Sequential(
            nn.Linear(3 * self.LINK_STATE_SIZE, self.FIRST_HIDDEN_LAYER_SIZE, device=device),
            nn.Tanh(),
            nn.Linear(self.FIRST_HIDDEN_LAYER_SIZE, self.FINAL_HIDDEN_LAYER_SIZE, device=device),
            nn.Tanh(),
            nn.Linear(self.FINAL_HIDDEN_LAYER_SIZE, self.LINK_STATE_SIZE, device=device),
            nn.Tanh(),
        )

        self.readout = nn.Sequential(
            nn.Linear(self.LINK_STATE_SIZE, self.FIRST_HIDDEN_LAYER_SIZE, device=device),
            nn.Tanh(),
            nn.Dropout(self.DROPOUT_RATE),
            nn.Linear(self.FIRST_HIDDEN_LAYER_SIZE, self.FINAL_HIDDEN_LAYER_SIZE, device=device),
            nn.Tanh(),
            nn.Dropout(self.DROPOUT_RATE),
            nn.Linear(self.FINAL_HIDDEN_LAYER_SIZE, self.NUM_ACTIONS, device=device),
        )

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                self.hidden_layer_initializer(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialize final layer differently if defined
        self.final_layer_initializer(self.readout[-1].weight)

    def message_passing(self, input: Batch) -> Tensor:

        link_states = input.x
        input_src, input_dst = input.edge_index  # corresponds to incoming and outgoing links
        link_states = F.pad(link_states, (0, self.LINK_STATE_SIZE - self.NUM_FEATURES))

        for _ in range(self.MESSAGE_ITERATIONS):
            incoming_link_states = link_states[input_src]  # [E, LINK_STATE_SIZE]
            outgoing_link_states = link_states[input_dst]  # [E, LINK_STATE_SIZE]
            message_inputs = torch.cat([incoming_link_states, outgoing_link_states], dim=1)
            messages = self.create_message(message_inputs)

            aggregated_messages = self.message_aggregation(messages, input_dst)
            link_update_input = torch.cat([link_states, aggregated_messages], dim=1)
            link_states = self.link_update(link_update_input)

        return link_states

    def message_aggregation(self, messages, outgoing_links):
        agg_max, _ = scatter_max(messages, outgoing_links, dim=0)
        agg_min, _ = scatter_min(messages, outgoing_links, dim=0)
        aggregated_messages = torch.cat([agg_max, agg_min], dim=1)
        return aggregated_messages

    def forward(self, input: Batch) -> Tensor:
        link_states = self.message_passing(input)
        policy = self.readout(link_states).view(-1)
        return policy
