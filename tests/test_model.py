"""Tests for model architecture and forward pass."""
import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.train import GINEdge


class TestGINEdge:
    @pytest.fixture
    def model(self):
        return GINEdge(hidden_dim=32, num_layers=2, dropout=0.3, pool_type='add')

    @pytest.fixture
    def sample_graph(self):
        """Create a minimal molecular graph (3 atoms, 2 bonds) with valid OGB vocab ranges."""
        # OGB AtomEncoder expects 9 dims with vocab sizes: [119, 5, 12, 12, 10, 6, 6, 2, 2]
        atom_vocab = [119, 5, 12, 12, 10, 6, 6, 2, 2]
        x = torch.stack([torch.randint(0, v, (3,)) for v in atom_vocab], dim=1)
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        # OGB BondEncoder expects 3 dims with vocab sizes: [5, 6, 2]
        bond_vocab = [5, 6, 2]
        edge_attr = torch.stack([torch.randint(0, v, (4,)) for v in bond_vocab], dim=1)
        batch = torch.zeros(3, dtype=torch.long)
        return x, edge_index, edge_attr, batch

    def test_forward_shape(self, model, sample_graph):
        x, ei, ea, batch = sample_graph
        out = model(x, ei, ea, batch)
        assert out.shape == (1,)

    def test_output_range(self, model, sample_graph):
        x, ei, ea, batch = sample_graph
        model.eval()
        with torch.no_grad():
            logit = model(x, ei, ea, batch)
            prob = torch.sigmoid(logit)
        assert 0 <= prob.item() <= 1

    def test_batch_forward(self, model):
        """Test forward with batch of 2 graphs."""
        atom_vocab = [119, 5, 12, 12, 10, 6, 6, 2, 2]
        x = torch.stack([torch.randint(0, v, (5,)) for v in atom_vocab], dim=1)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 4]], dtype=torch.long)
        bond_vocab = [5, 6, 2]
        edge_attr = torch.stack([torch.randint(0, v, (4,)) for v in bond_vocab], dim=1)
        batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
        out = model(x, edge_index, edge_attr, batch)
        assert out.shape == (2,)

    def test_pool_types(self, sample_graph):
        x, ei, ea, batch = sample_graph
        for pool in ['add', 'mean']:
            m = GINEdge(hidden_dim=32, num_layers=2, dropout=0.3, pool_type=pool)
            out = m(x, ei, ea, batch)
            assert out.shape == (1,)

    def test_gradient_flow(self, model, sample_graph):
        x, ei, ea, batch = sample_graph
        model.train()
        out = model(x, ei, ea, batch)
        loss = out.sum()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters() if p.requires_grad)
        assert has_grad

    def test_param_count(self, model):
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0
        assert n_params < 5_000_000  # should be relatively small
