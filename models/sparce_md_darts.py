import torch
import torch.nn as nn
from typing import Optional, List
from nni.retiarii.nn.pytorch import LayerChoice, InputChoice
from collections import OrderedDict
import json

from models.md_darts import CNN


class MdDartsSparceLayerChoice(nn.Module):
    def __init__(self, layer_choice, domain_to_name: List[str]):
        """
        param: layer_choice: len of layer_choice <= number of domains
        param: domain_to_name: list of len (num_domains)
        """
        super(MdDartsSparceLayerChoice, self).__init__()
        self.name = layer_choice.label
        self.op_choices = nn.ModuleDict(OrderedDict([(name, layer_choice[name]) for name in layer_choice.names]))
        self.domain_to_name = domain_to_name

    def forward(self, inputs, domain_idx: int):
        output = self.op_choices[self.domain_to_name[domain_idx]](inputs, domain_idx)
        return output


class MdDartsSparceInputChoice(nn.Module):
    def __init__(self, input_choice, domain_to_chosen: List[List[int]]):
        """
        param: input_choice: input choice
        param: domain_to_chosen: list of chosen indices for each domain
        """
        super(MdDartsSparceInputChoice, self).__init__()
        self.name = input_choice.label
        self.n_chosen = input_choice.n_chosen or 1
        assert len(domain_to_chosen[0]) == self.n_chosen
        self.domain_to_chosen = domain_to_chosen

    def forward(self, inputs, domain_idx: int):
        inputs = torch.stack(inputs)
        output = inputs[self.domain_to_chosen[domain_idx]].mean(dim=0)
        return output


class SparceMdDartsModel(nn.Module):
    def __init__(self, config, checkpoint_path: Optional[str] = None):
        super(SparceMdDartsModel, self).__init__()
        self.model = CNN(int(config['darts']['input_size']),
                         int(config['darts']['input_channels']),
                         int(config['darts']['channels']),
                         int(config['darts']['n_classes']),
                         int(config['darts']['layers']),
                         n_heads=len(config['datasets'].split(';')),
                         n_nodes=int(config['darts']['n_nodes']),
                         stem_multiplier=int(config['darts']['stem_multiplier']))

        self.architectures = json.loads(open(config['architecture_path']).read())

        # initialize sparce model
        def apply_layer_choice(m):
            for name, child in m.named_children():
                if isinstance(child, LayerChoice):
                    setattr(m, name, MdDartsSparceLayerChoice(child, [a[child.key] for a in self.architectures]))
                else:
                    apply_layer_choice(child)

        def apply_input_choice(m):
            for name, child in m.named_children():
                if isinstance(child, InputChoice):
                    setattr(m, name, MdDartsSparceInputChoice(child, [a[child.key] for a in self.architectures]))
                else:
                    apply_input_choice(child)

        apply_layer_choice(self.model)
        apply_input_choice(self.model)

        if checkpoint_path is not None:
            self.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['model_state_dict'])

    def forward(self, X: torch.tensor, domain_idx: int):
        return self.model(X, domain_idx)
