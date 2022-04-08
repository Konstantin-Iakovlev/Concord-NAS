import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class Controller(nn.Module):
    def __init__(self, n_layers, nodes_per_layer, primitives_names: List[str], dim_hidden=32, num_domains=2,
                 ma_decay=0.95, learning_rate=3e-4):
        super(Controller, self).__init__()
        self.n_layers = n_layers
        self.nodes_per_layer = nodes_per_layer
        self.num_domains = num_domains
        self.primitives_names = primitives_names
        self.primitives_to_idx = {p: i for i, p in enumerate(primitives_names)}

        self.encoder_prim = nn.Embedding(len(primitives_names), dim_hidden)
        self.encoder_pos = nn.Embedding(nodes_per_layer, dim_hidden)

        self.lstm = nn.LSTMCell(dim_hidden, dim_hidden)
        self.decoders = nn.ModuleDict({
            'primitives': nn.Linear(dim_hidden, len(primitives_names)),
            'positions': nn.Linear(dim_hidden, nodes_per_layer),
        })
        self.h0 = nn.Parameter(torch.zeros(num_domains, dim_hidden))
        self.c0 = nn.Parameter(torch.zeros(num_domains, dim_hidden))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.baseline = None
        self.decay = ma_decay

    def forward(self, path: List[List[Tuple[int, int, str]]]) -> torch.Tensor:
        """
        param: path: network path
        returns: log probabilities of shape (num_domains)
        """
        h_curr, c_curr = self.h0, self.c0
        input_states = []  # (seq_len, num_domains)
        inp_state = torch.LongTensor([path[d][0][0] for d in range(len(path))]).to(self.h0.device)
        input_states.append(inp_state)
        for layer_idx in range(len(path[0])):
            # add primitive
            inp_state = torch.LongTensor([self.primitives_to_idx[path[d][layer_idx][2]] for d in range(len(path))]).to(self.h0.device)
            input_states.append(inp_state)

            # add position
            inp_state = torch.LongTensor([path[d][layer_idx][1] for d in range(len(path))]).to(self.h0.device)
            input_states.append(inp_state)

        log_probs = torch.zeros(self.num_domains, requires_grad=True).to(self.h0.device)
        pos_logits = self.decoders['positions'](h_curr)
        dist = torch.distributions.Categorical(logits=pos_logits)
        log_probs = log_probs + dist.log_prob(input_states[0])

        for i in range(len(input_states) - 1):
            if i % 2 == 0:
                h_curr, c_curr = self.lstm(self.encoder_pos(input_states[i]), (h_curr, c_curr))
                logits = self.decoders['primitives'](h_curr)
            else:
                h_curr, c_curr = self.lstm(self.encoder_prim(input_states[i]), (h_curr, c_curr))
                logits = self.decoders['positions'](h_curr)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs += dist.log_prob(input_states[i + 1])

        return log_probs

    @torch.no_grad()
    def sample(self, greedy=False) -> List[List[Tuple[int, int, str]]]:
        """
        param: greedy: if True then use argmax strategy, otherwise sampling
        returns: network paths for each domain
        """
        h_curr, c_curr = self.h0, self.c0
        path: List[List[int, int, str]] = []
        positions = []
        primitives = []

        from_logits = self.decoders['positions'](h_curr)
        dist = torch.distributions.Categorical(logits=from_logits)
        if greedy:
            from_ids = dist.sample()
        else:
            from_ids = from_logits.argmax(-1)

        positions.append(from_ids)
        for i in range(self.n_layers):
            # generate primitive
            h_curr, c_curr = self.lstm(self.encoder_pos(positions[-1]), (h_curr, c_curr))
            prim_logits = self.decoders['primitives'](h_curr)
            dist = torch.distributions.Categorical(logits=prim_logits)
            if not greedy:
                prim_ids = dist.sample()
            else:
                prim_ids = prim_logits.argmax(-1)
            primitives.append(prim_ids)

            # generate position
            h_curr, c_curr = self.lstm(self.encoder_prim(primitives[-1]), (h_curr, c_curr))
            pos_logits = self.decoders['positions'](h_curr)
            dist = torch.distributions.Categorical(logits=pos_logits)
            if not greedy:
                pos_ids = dist.sample()
            else:
                pos_ids = pos_logits.argmax(-1)
            positions.append(pos_ids)

        # construct path
        for domain_idx in range(self.num_domains):
            path_domain: List[Tuple[int, int, str]] = []
            for i in range(len(positions) - 1):
                path_domain.append((positions[i][domain_idx].item(), positions[i + 1][domain_idx].item(),
                                    self.primitives_names[primitives[i][domain_idx].item()]))
            path.append(path_domain)

        return path

    def update(self, reward: torch.Tensor, path: List[List[Tuple[int, int, str]]]):
        """
        param: reward: tensor of shape (num_domains)
        param: path: network path that gives a reward
        """
        if self.baseline is None:
            self.baseline = reward
        # EMA baseline
        self.baseline = self.decay * self.baseline + (1 - self.decay) * reward

        self.optimizer.zero_grad()
        loss = ((reward - self.baseline) * self.forward(path)).sum()
        # maximize
        (-loss).backward()
        self.optimizer.step()
