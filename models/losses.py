import torch
import torch.nn as nn


class MdTripletLoss(nn.Module):
    def __init__(self):
        super(MdTripletLoss, self).__init__()
        self.distance = nn.PairwiseDistance()
        # TODO: adjust margin
        self.triplet_loss = nn.TripletMarginLoss(margin=0.0)

    def forward(self, h1: torch.Tensor, h2: torch.Tensor, labels1: torch.LongTensor, labels2: torch.LongTensor):
        """
        :param: h1: hidden representations of size (bs, *), anchors
        :param: h1: hidden representations of size (bs, *), positives and negatives candidates
        """
        bs = h1.size(0)
        dist_matrix = self.distance(h1.view(bs, 1, -1).broadcast_to(bs, bs, -1).reshape(bs * bs, -1),
                                    h2.view(1, bs, -1).broadcast_to(bs, bs, -1).reshape(bs * bs, -1)).view(bs, bs)
        label_matrix = (labels1.view(-1, 1) == labels2.view(1, -1))
        valid_triples = label_matrix.any(-1) & (~label_matrix).any(-1)

        if torch.all(~valid_triples).item():
            return torch.tensor(0.0).to(h1.device).requires_grad_()

        dist_matrix = dist_matrix[valid_triples]
        label_matrix = label_matrix[valid_triples]

        hard_pos_ids = dist_matrix.masked_fill(~label_matrix, -float('inf')).argmax(-1)
        hard_neg_ids = dist_matrix.masked_fill(label_matrix, float('inf')).argmin(-1)
        return self.triplet_loss(h1.view(bs, -1)[valid_triples],
                                 h2.view(bs, -1)[hard_pos_ids],
                                 h2.view(bs, -1)[hard_neg_ids])
