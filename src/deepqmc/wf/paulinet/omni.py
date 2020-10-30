from collections import defaultdict
from functools import partial

from torch import nn

from .backflow import Backflow
from .distbasis import DistanceBasis
from .jastrow import Jastrow
from .schnet import ElectronicSchNet


class OmniCore(nn.Module):
    def __init__(
        self,
        mol,
        n_up,
        n_down,
        backflow_spec,
        *,
        dist_feat_dim=32,
        dist_feat_cutoff=10.0,
        many_body_dnn_factory=ElectronicSchNet,
        mean_field_dnn_factory=partial(ElectronicSchNet, version='mean-field'),
        jastrow_factory=Jastrow,
        jastrow_stream='many_body_emb',
        backflow_factory=Backflow,
        backflow_stream='many_body_emb',
        many_body_dnn_kwargs={},
        mean_field_dnn_kwargs={},
        jastrow_kwargs={},
        backflow_kwargs={},
    ):

        super().__init__()
        self.dnn_core = nn.ModuleDict(
            {
                'many-body': many_body_dnn_factory(
                    n_up,
                    n_down,
                    len(mol.coords),
                    dist_feat_dim=dist_feat_dim,
                    **many_body_dnn_kwargs,
                )
                if many_body_dnn_factory
                else None,
                'mean-field': mean_field_dnn_factory(
                    n_up,
                    n_down,
                    len(mol.coords),
                    dist_feat_dim=dist_feat_dim,
                    **mean_field_dnn_kwargs,
                )
                if mean_field_dnn_factory
                else None,
            }
        )
        self.dist_basis = (
            DistanceBasis(dist_feat_dim, cutoff=dist_feat_cutoff, envelope='nocusp')
            if 'edges' in jastrow_stream + backflow_stream
            or 'emb' in jastrow_stream + backflow_stream
            else None
        )
        self.streams = {
            'dists_nuc': (self.dists_nuc_stream, 1),
            'dists_elec': (self.dists_elec_stream, 1),
            'edges_nuc': (self.edges_nuc_stream, dist_feat_dim)
            if self.dist_basis
            else (None, None),
            'edges_elec': (self.edges_elec_stream, dist_feat_dim)
            if self.dist_basis
            else (None, None),
            'many_body_emb': (
                self.many_body_emb_stream,
                self.dnn_core['many-body'].embedding_dim,
            )
            if self.dnn_core['many-body']
            else (None, None),
            'mean_field_emb': (
                self.mean_field_emb_stream,
                self.dnn_core['mean-field'].embedding_dim,
            )
            if self.dnn_core['mean-field']
            else (None, None),
        }
        self.jastrow_stream = self.streams[jastrow_stream][0]
        assert not jastrow_factory or self.jastrow_stream
        self.jastrow = (
            jastrow_factory(self.streams[jastrow_stream][1], **jastrow_kwargs)
            if jastrow_factory
            else None
        )
        self.backflow_stream = self.streams[backflow_stream][0]
        assert not backflow_factory or self.backflow_stream
        self.backflow = (
            backflow_factory(
                self.streams[backflow_stream][1], *backflow_spec, **backflow_kwargs
            )
            if backflow_factory
            else None
        )
        self.memory_reset()
        # if freeze_embed:
        #    self.requires_grad_embeddings_(False)

    def memory_check(self, dists_elec, dists_nuc):
        if not self.memory_id:
            self.memory_id = id(dists_elec) + id(dists_nuc)
        elif self.memory_id != id(dists_elec) + id(dists_nuc):
            self.memory_reset()

    def memory_reset(self):
        self.memory = defaultdict(lambda: None, {})
        self.memory_id = None

    def dists_nuc_stream(self, dists_nuc, dists_elec):
        if self.memory['dists_nuc'] is None:
            self.memory['dists_nuc'] = dists_nuc
        return self.memory['dists_nuc']

    def dists_elec_stream(self, dists_nuc, dists_elec):
        if self.memory['dists_elec'] is None:
            self.memory['dists_elec'] = dists_elec
        return self.memory['dists_elec']

    def edges_nuc_stream(self, dists_nuc, dists_elec):
        if self.memory['edges_nuc'] is None:
            self.memory['edges_nuc'] = self.dist_basis(
                self.dists_nuc_stream(dists_nuc, dists_elec)
            )
        return self.memory['edges_nuc']

    def edges_elec_stream(self, dists_nuc, dists_elec):
        if self.memory['edges_elec'] is None:
            self.memory['edges_elec'] = self.dist_basis(
                self.dists_elec_stream(dists_nuc, dists_elec)
            )
        return self.memory['edges_elec']

    def many_body_emb_stream(self, dists_nuc, dists_elec):
        if self.memory['many_body_emb'] is None:
            self.memory['many_body_emb'] = self.dnn_core['many-body'](
                self.edges_elec_stream(dists_nuc, dists_elec),
                self.edges_nuc_stream(dists_nuc, dists_elec),
            )
        return self.memory['many_body_emb']

    def mean_field_emb_stream(self, dists_nuc, dists_elec):
        if self.memory['mean_field_emb'] is None:
            self.memory['mean_field_emb'] = self.dnn_core['mean-field'](
                self.edges_elec_stream(dists_nuc, dists_elec),
                self.edges_nuc_stream(dists_nuc, dists_elec),
            )
        return self.memory['mean_field_emb']

    def forward_jastrow(self, dists_nuc, dists_elec):
        self.memory_check(dists_nuc, dists_elec)
        return self.jastrow(self.jastrow_stream(dists_nuc, dists_elec))

    def forward_backflow(self, dists_nuc, dists_elec):
        self.memory_check(dists_nuc, dists_elec)
        return self.backflow(self.backflow_stream(dists_nuc, dists_elec))
