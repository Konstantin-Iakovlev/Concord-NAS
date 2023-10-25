import torch
import torch.nn as nn
from .ops import ConvBN, Pool, Zero, Identity
from torch.distributions import RelaxedOneHotCategorical

# TODO: share structural parameters
# TODO: attention maks conflict when using is_pair = True


def ops_factory(op_name: str, channels: int):
    if op_name == 'maxpool':
        return Pool('max', 3, 1)
    elif op_name == 'avgpool':
        return Pool('avg', 3, 1)
    elif op_name == 'skipconnect':
        return Identity()
    elif op_name == 'conv3x3':
        return ConvBN(channels, 3)
    elif op_name == 'conv5x5':
        return ConvBN(channels, 5)
    elif op_name == 'conv7x7':
        return ConvBN(channels, 7)
    elif op_name == 'dilconv3x3':
        return ConvBN(channels, 3, dilation=True)
    elif op_name == 'dilconv5x5':
        return ConvBN(channels, 5, dilation=True)
    elif op_name == 'dilconv7x7':
        return ConvBN(channels, 7, dilation=True)
    elif op_name == 'zero':
        return Zero()
    else:
        raise ValueError(f'Unknown operation {op_name}')


class LayerChoice(nn.Module):
    def __init__(self, channels, label='none') -> None:
        super().__init__()
        self.op_names = ('maxpool', 'avgpool', 'skipconnect', 'conv3x3', 'conv5x5',
                         'conv7x7', 'dilconv3x3', 'dilconv5x5', 'dilconv7x7', 'zero')
        self.label = label
        self.ops = nn.ModuleList([
            ops_factory(op, channels) for op in self.op_names
        ])
        self.alpha = nn.Parameter(torch.randn(len(self.op_names)) * 1e-3)
    
    def forward(self, x: torch.Tensor, msk: torch.Tensor):
        mixed_out = torch.stack([op(x, msk) for op in self.ops], 0)
        weights = RelaxedOneHotCategorical(logits=self.alpha, temperature=0.1).rsample().reshape(-1, 1, 1, 1)
        out = (weights * mixed_out).sum(0)
        return out


class OneHotLayerChoice(nn.Module):
    def __init__(self, channels, op_name, label='none') -> None:
        super().__init__()
        self.label = label
        self.op = ops_factory(op_name, channels)
    
    def forward(self, x: torch.Tensor, msk: torch.Tensor):
        return self.op(x, msk)


def test_lc():
    lc = LayerChoice(128)
    h = torch.randn(1, 32, 128)
    msk = torch.ones(1, 32)
    out = lc(h, msk)
    assert out.shape == h.shape


def test_oh_lc():
    lc = OneHotLayerChoice(128, 'dilconv5x5')
    h = torch.randn(1, 32, 128)
    msk = torch.ones(1, 32)
    out = lc(h, msk)
    assert out.shape == h.shape


class InputSwitch(nn.Module):
    def __init__(self, n_cand, n_chosen, label='none', genotype=None) -> None:
        super().__init__()
        self.n_chosen = n_chosen
        self.label = label
        self.n_cand = n_cand
        if genotype is None:
            self.alpha = nn.Parameter(torch.randn(n_cand) * 1e-3)
        else:
            self.register_buffer('alpha', torch.zeros(n_cand))
    
    def forward(self, inputs: torch.Tensor):
        """inputs: (n_cand, bs, seq_len, hidden)"""
        weights = self.alpha.softmax(-1).reshape(-1, 1, 1, 1)
        return (weights * inputs).sum(0)


class Node(nn.Module):
    def __init__(self, node_id, num_prev_nodes, channels, genotype=None) -> None:
        """genotype: [(op, prev_node_idx)]"""
        super().__init__()
        self.edges = nn.ModuleDict([])
        for i in range(num_prev_nodes):
            if genotype is None:
                self.edges.update({f'{i}': LayerChoice(channels, f'{node_id}_p{i}')})
            elif i in [e[1] for e in genotype]:
                op_name = None
                for o, j in genotype:
                    if j == i:
                        op_name = o
                self.edges.update({f'{i}': OneHotLayerChoice(channels, op_name, f'{node_id}_p{i}')})
        self.input_switch = InputSwitch(len(self.edges), 2, f'{node_id}_switch', genotype)
    
    def forward(self, prev_nodes: torch.Tensor, msk: torch.Tensor):
        """prev_nodes: List (bs, seq_len, hidden)"""
        res = []
        for key, op in self.edges.items():
            res.append(op(prev_nodes[int(key)], msk))
        return self.input_switch(torch.stack(res, 0))


def test_node():
    node = Node(3, 3, 128, None)
    h = torch.randn(1, 32, 128)
    prev_nodes = [h, h, h]
    msk = torch.ones(1, 32)
    out = node(prev_nodes, msk)
    assert out.shape == h.shape


class Cell(nn.Module):
    def __init__(self, n_nodes, channels, dropout, genotype=None) -> None:
        """Genotype: list of genotypes for each node"""
        super().__init__()
        self.preprocess = nn.ModuleList([nn.Sequential(nn.LayerNorm(channels), nn.Dropout(dropout)) for _ in range(2)])
        nodes = []
        for depth in range(2, n_nodes + 2):
            nodes.append(Node(f'n{depth}', depth, channels, genotype[depth - 2] if genotype is not None else genotype))
        self.nodes = nn.ModuleList(nodes)
        self.att_weights = nn.Parameter(torch.randn(n_nodes) * 1e-3)
    
    def forward(self, s0, s1, msk):
        inputs = [self.preprocess[0](s0), self.preprocess[1](s1)]
        for node in self.nodes:
            out = node(inputs, msk)
            inputs.append(out)
        out = torch.stack(inputs[2:], dim=0)  # (n_nodes, bs, seq_len, hidden)
        weights = torch.softmax(self.att_weights, -1).reshape(-1, 1, 1, 1)
        return (out * weights).sum(0)


def test_cell():
    cell = Cell(3, 128, 0.1)
    s0 = s1 = torch.randn(1, 32, 128)
    msk = torch.ones(1, 32)
    out = cell(s0, s1, msk)
    assert out.shape == (1, 32, 128, 3)


class AdaBertStudent(nn.Module):
    def __init__(self,
                 vocab_size,
                 is_pair_task,
                 num_classes,
                 num_interm_nodes = 3,
                 num_cells = 8,
                 emb_size = 128,
                 dropout_p = 0.1,
                 genotype = None,
                 ) -> None:
        """genotype: List[List[(op, id)]]"""
        super().__init__()
        self.token_embeds = nn.Embedding(vocab_size, emb_size)
        emb_matrix = torch.load('low_rank_emb.pth')
        self.token_embeds.weight.data = emb_matrix.data
        cells = []
        for i in range(num_cells):
            cell = Cell(num_interm_nodes, emb_size, dropout_p, genotype)
            # TODO: share parameters here
            if i > 0 and genotype is None:
                for node, ref_node in zip(cell.nodes, cells[-1].nodes):
                    assert node.edges.keys() == ref_node.edges.keys()
                    for key in node.edges:
                        node.edges[key].alpha = ref_node.edges[key].alpha
                    node.input_switch.alpha = ref_node.input_switch.alpha
            cells.append(cell)
        self.cells = nn.ModuleList(cells)
        
        self.mlp = nn.Sequential(nn.Tanh(), nn.Dropout(dropout_p), nn.Linear(emb_size, num_classes))
        self.is_pair_task = is_pair_task
    
    def forward(self, ids: torch.LongTensor, msk: torch.Tensor):
        x = self.token_embeds(ids)
        if self.is_pair_task:
            s0, s1 = x[0], x[1]
        else:
            s0 = s1 = x
        for _, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, msk)
        out = s1.mean(1)  # (bs, hidden)
        logits = self.mlp(out)
        return logits


def test_bert():
    genotype = [[('conv7x7', 0), ('maxpool', 1)],
                [('maxpool', 1), ('maxpool', 2)],
                [('conv3x3', 1), ('dilconv3x3', 3)]]
    bert = AdaBertStudent(30_522, False, 2, genotype=None)
    ids = torch.zeros(3, 32).long()
    msk = torch.ones(3, 32)
    out = bert(ids, msk)
    print("total params", sum([p.numel() for p in bert.parameters()]))
    assert out.shape == (3, 2)