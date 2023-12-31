import torch
import torch.nn as nn
from .ops import ConvBN, Pool, Zero, Identity
from torch.distributions import RelaxedOneHotCategorical
import numpy as np

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
    def __init__(self, channels, num_domains, label='none') -> None:
        super().__init__()
        self.op_names = ('maxpool', 'avgpool', 'skipconnect', 'conv3x3', 'conv5x5',
                         'conv7x7', 'dilconv3x3', 'dilconv5x5', 'dilconv7x7')#, 'zero')
        self.label = label
        self.ops = nn.ModuleList([
            ops_factory(op, channels) for op in self.op_names
        ])
        self.alpha = nn.Parameter(torch.randn(num_domains, len(self.op_names)) * 1e-3)
        self.temperature = 1.0
    
    def forward(self, x: torch.Tensor, msk: torch.Tensor, domain_idx) -> torch.Tensor:
        mixed_out = torch.stack([op(x, msk) for op in self.ops], 0)
        weights = RelaxedOneHotCategorical(logits=self.alpha[domain_idx],
                                           temperature=self.temperature).rsample().reshape(-1, 1, 1, 1)
        # weights = self.alpha.softmax(-1).reshape(-1, 1, 1, 1)
        out = (weights * mixed_out).sum(0)
        return out
    
    def export(self, domain_idx: int):
        return self.op_names[self.alpha[domain_idx].argmax(-1).item()]


class OneHotLayerChoice(nn.Module):
    def __init__(self, channels, op_names_to_domains, num_domains, label='none') -> None:
        """op_names_to_domains: {op_name: [d1, d2, ...]}"""
        super().__init__()
        self.label = label
        self.ops = nn.ModuleList([ops_factory(op_name, channels) for op_name in op_names_to_domains])
        self.domain_to_op = [0] * num_domains
        for i, domains in enumerate(op_names_to_domains.values()):
            for d in domains:
                self.domain_to_op[d] = i
    
    def forward(self, x: torch.Tensor, msk: torch.Tensor, domain_idx: int) -> torch.Tensor:
        return self.ops[self.domain_to_op[domain_idx]](x, msk)


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
    def __init__(self, n_cand, n_chosen, num_domains, label='none', genotype=None) -> None:
        super().__init__()
        self.n_chosen = n_chosen
        self.label = label
        self.n_cand = n_cand
        if genotype is None:
            self.alpha = nn.Parameter(torch.randn(num_domains, n_cand) * 1e-3)
        else:
            self.register_buffer('alpha', torch.zeros(num_domains, n_cand))
        self.genotype = genotype
        self.temperature = 1.0
    
    def forward(self, inputs: torch.Tensor, domain_idx: int) -> torch.Tensor:
        """inputs: (n_cand, bs, seq_len, hidden)"""
        if self.genotype is None:
            weights = RelaxedOneHotCategorical(logits=self.alpha[domain_idx],
                                               temperature=self.temperature).rsample().reshape(-1, 1, 1, 1)
        else:
            weights = self.alpha[domain_idx].softmax(-1).reshape(-1, 1, 1, 1)
        return (inputs * weights).sum(0)
    
    def export(self, domain_idx: int):
        return np.argsort(self.alpha[domain_idx].detach().cpu().numpy())[-2:].tolist()


class Node(nn.Module):
    def __init__(self, node_id, num_prev_nodes, channels, num_domains, genotype=None) -> None:
        """genotype: [(op, prev_node_idx)] for each domain"""
        super().__init__()
        print('Node', genotype)
        self.edges = nn.ModuleDict([])
        for i in range(num_prev_nodes):
            if genotype is None:
                self.edges.update({f'{i}': LayerChoice(channels, num_domains, f'{node_id}_p{i}')})
            elif i in [e[1] for gen in genotype for e in gen]:
                op_to_domains = {}
                for d, gen in enumerate(genotype):
                    for o, j in gen:
                        if j == i:
                            if o not in op_to_domains:
                                op_to_domains[o] = [d]
                            else:
                                op_to_domains[o].append(d)

                print(op_to_domains)
                self.edges.update({f'{i}': OneHotLayerChoice(channels, op_to_domains, num_domains, f'{node_id}_p{i}')})
        self.input_switch = InputSwitch(num_prev_nodes if genotype is None else 2, 2, num_domains,
                                        f'{node_id}_switch', genotype)
        self.num_domains = num_domains
    
    def forward(self, prev_nodes: torch.Tensor, msk: torch.Tensor, domain_idx: int):
        """prev_nodes: List (bs, seq_len, hidden)"""
        res = []
        for key, op in self.edges.items():
            res.append(op(prev_nodes[int(key)], msk, domain_idx))
        return self.input_switch(torch.stack(res, 0), domain_idx)
    
    def export(self, domain_idx):
        selected_edges = self.input_switch.export(domain_idx)
        selected_ops = [self.edges[f'{i}'].export(domain_idx) for i in selected_edges]
        return [[op, i] for op, i in zip(selected_ops, selected_edges)]


def test_node():
    node = Node(3, 3, 128, None)
    h = torch.randn(1, 32, 128)
    prev_nodes = [h, h, h]
    msk = torch.ones(1, 32)
    out = node(prev_nodes, msk)
    assert out.shape == h.shape


class Cell(nn.Module):
    def __init__(self, n_nodes, channels, dropout, num_domains, genotype=None) -> None:
        """Genotype: list of genotypes for each node"""
        super().__init__()
        self.preprocess = nn.ModuleList([nn.Sequential(nn.LayerNorm(channels), nn.Dropout(dropout)) for _ in range(2)])
        nodes = []
        for depth in range(2, n_nodes + 2):
            nodes.append(Node(f'n{depth}', depth, channels, num_domains, [gen[depth - 2] for gen in genotype] \
                              if genotype is not None else genotype))
        self.nodes = nn.ModuleList(nodes)
        self.att_weights = nn.Parameter(torch.randn(n_nodes) * 1e-3)
    
    def forward(self, s0, s1, msk, domain_idx):
        inputs = [self.preprocess[0](s0), self.preprocess[1](s1)]
        for node in self.nodes:
            out = node(inputs, msk, domain_idx)
            inputs.append(out)
        out = torch.stack(inputs[2:], dim=0)  # (n_nodes, bs, seq_len, hidden)
        out = (out * self.att_weights.reshape(-1, 1, 1, 1)).sum(0)
        return out
    
    def export(self, domain_idx: int):
        return [n.export(domain_idx) for n in self.nodes]


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
                 num_domains,
                 pretrained_token_embeddings,
                 pretrained_pos_embeddings,
                 num_interm_nodes = 3,
                 num_cells = 8,
                 emb_size = 128,
                 dropout_p = 0.1,
                 genotype = None,
                 ) -> None:
        """genotype: List[List[(op, id)]]"""
        super().__init__()
        self.token_embeds = nn.Embedding(vocab_size, pretrained_token_embeddings.shape[1])
        self.token_embeds.weight = pretrained_token_embeddings
        self.token_embeds.requires_grad = False # freeze pretrained embeddings
        self.pos_embeds = nn.Embedding(pretrained_pos_embeddings.shape[0], pretrained_pos_embeddings.shape[1])
        self.pos_embeds.weight = pretrained_pos_embeddings
        self.pos_embeds.weight.requires_grad = False
        self.fact_map = nn.Linear(pretrained_token_embeddings.shape[1], emb_size)
        self.type_embeds = nn.Embedding(2, emb_size)
        self.num_domains = num_domains
        cells = []
        for i in range(num_cells):
            cell = Cell(num_interm_nodes, emb_size, dropout_p, num_domains, genotype)
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
        self.emb_dropout = nn.Dropout(dropout_p)
    
    def forward(self, ids: torch.LongTensor, type_ids: torch.LongTensor, msk: torch.Tensor, domain_idx: int):
        """Pefrorms a forward pass

        :param ids: tensor of shape (sentences, batch_size, seq_len)
        :param type_ids: tensor of shape (sentences, batch_size, seq_len)
        :param msk: tensor of shape (sentences, batch_size, seq_len)
        :return: tensor of shape (bs)
        """
        pos_ids = torch.arange(ids.shape[2])[None, None].broadcast_to(ids.shape).to(ids.device)
        x = self.fact_map(self.token_embeds(ids) + self.pos_embeds(pos_ids)) + self.type_embeds(type_ids)
        x = self.emb_dropout(x)
        if self.is_pair_task:
            s0, s1 = x[0], x[1]
        else:
            s0 = s1 = x[0]
        for _, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, msk, domain_idx)
        out = s1.mean(1)  # (bs, hidden)
        logits = self.mlp(out)
        return logits
    
    def export(self):
        return [self.cells[0].export(d) for d in range(self.num_domains)]
    
    def set_temperature(self, tau: float):
        for m in self.modules():
            if hasattr(m, 'temperature'):
                setattr(m, 'temperature', tau)


def test_bert():
    genotype = [[('conv7x7', 0), ('maxpool', 1)],
                [('maxpool', 1), ('maxpool', 2)],
                [('conv3x3', 1), ('dilconv3x3', 3)]]
    pret_embed = nn.Parameter(torch.randn(30_522, 768))
    pret_pos = nn.Parameter(torch.randn(512, 768))
    bert = AdaBertStudent(30_522, False, 2, pret_embed, pret_pos, genotype=None)
    ids = torch.zeros(3, 32).long()
    type_ids = torch.zeros(3, 32).long()
    msk = torch.ones(3, 32)
    out = bert(ids, type_ids, msk)
    print("total params", sum([p.numel() for p in bert.parameters()]))
    assert out.shape == (3, 2)
    assert bert.cells[0].export() == bert.cells[1].export()


def distil_loss(pi_logits: torch.Tensor, p_scores: torch.Tensor):
    pi_probs = pi_logits.softmax(-1)
    return -(pi_probs * torch.log_softmax(p_scores, -1)).sum(-1).mean()


def evaluate(model, dl, device):
    model.eval()
    n_total = 0
    n_corr = 0
    for batch in dl:
        batch = {k: batch[k].to(device) for k in batch}
        pi_logits = batch['logits']
        inp_ids = batch['inp_ids']
        type_ids = batch['type_ids']
        msk = batch['att'].max(0).values
        with torch.no_grad():
            p_logits = model(inp_ids, type_ids, msk, 0)
        n_total += p_logits.shape[0]
        n_corr += (pi_logits.argmax(-1) == p_logits.argmax(-1)).sum().item()
    return n_corr / n_total
