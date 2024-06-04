import torch
import torch.nn as nn
import pygmtools as pygm
from pygmtools.linear_solvers import sinkhorn
# from decomposition import get_zero_sum_vectors, pair_decomposition, get_iso_matrix
# from pygmtools.utils import permutation_loss
import lightning as L
pygm.set_backend('pytorch')
# from src.lap_solvers.sinkhorn import Sinkhorn, GumbelSinkhorn
# from src.build_graphs import reshape_edge_feature
# from src.feature_align import feature_align
# from src.factorize_graph_matching import construct_aff_mat
from models.NGM.gnn import GNNLayer
from models.NGM.geo_edge_feature import geo_edge_feature
# from models.GMN.affinity_layer import InnerpAffinity, GaussianAffinity
from src.evaluation_metric import objective_score
# from src.lap_solvers.hungarian import hungarian
import math
from src.utils.gpu_memory import gpu_free_memory
#from torch_geometric.data import Data, Batch
#from torch_geometric.utils import dense_to_sparse, to_dense_batch
# from src.utils.config import cfg
# from src.backbone import *
CNN = eval(cfg.BACKBONE)


class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        if cfg.NGM.EDGE_FEATURE == 'cat':
            self.affinity_layer = InnerpAffinity(cfg.NGM.FEATURE_CHANNEL)
        elif cfg.NGM.EDGE_FEATURE == 'geo':
            self.affinity_layer = GaussianAffinity(1, cfg.NGM.GAUSSIAN_SIGMA)
        else:
            raise ValueError('Unknown edge feature type {}'.format(cfg.NGM.EDGE_FEATURE))

    def forward(self, data_dict, **kwargs):
        batch_size = data_dict['batch_size']
        if 'images' in data_dict:
            # real image data
            src, tgt = data_dict['images']
            P_src, P_tgt = data_dict['Ps']
            ns_src, ns_tgt = data_dict['ns']
            G_src, G_tgt = data_dict['Gs']
            H_src, H_tgt = data_dict['Hs']
            K_G, K_H = data_dict['KGHs']

            # extract feature
            src_node = self.node_layers(src)
            src_edge = self.edge_layers(src_node)
            tgt_node = self.node_layers(tgt)
            tgt_edge = self.edge_layers(tgt_node)

            # feature normalization
            src_node = self.l2norm(src_node)
            src_edge = self.l2norm(src_edge)
            tgt_node = self.l2norm(tgt_node)
            tgt_edge = self.l2norm(tgt_edge)

            # arrange features
            U_src = feature_align(src_node, P_src, ns_src, self.rescale)
            F_src = feature_align(src_edge, P_src, ns_src, self.rescale)
            U_tgt = feature_align(tgt_node, P_tgt, ns_tgt, self.rescale)
            F_tgt = feature_align(tgt_edge, P_tgt, ns_tgt, self.rescale)
        elif 'features' in data_dict:
            # synthetic data
            src, tgt = data_dict['features']
            P_src, P_tgt = data_dict['Ps']
            ns_src, ns_tgt = data_dict['ns']
            G_src, G_tgt = data_dict['Gs']
            H_src, H_tgt = data_dict['Hs']
            K_G, K_H = data_dict['KGHs']

            U_src = src[:, :src.shape[1] // 2, :]
            F_src = src[:, src.shape[1] // 2:, :]
            U_tgt = tgt[:, :tgt.shape[1] // 2, :]
            F_tgt = tgt[:, tgt.shape[1] // 2:, :]
        elif 'aff_mat' in data_dict:
            K = data_dict['aff_mat']
            ns_src, ns_tgt = data_dict['ns']
        else:
            raise ValueError('Unknown data type for this model.')

        X = reshape_edge_feature(F_src, G_src, H_src)
        Y = reshape_edge_feature(F_tgt, G_tgt, H_tgt)

        # affinity layer
        Me, Mp = self.affinity_layer(X, Y, U_src, U_tgt)

        M = construct_aff_mat(Me, Mp, K_G, K_H)

        v = self.gm_solver(M, num_src=P_src.shape[1], ns_src=ns_src, ns_tgt=ns_tgt)
        s = v.view(v.shape[0], P_tgt.shape[1], -1).transpose(1, 2)

        s = self.sinkhorn(s, ns_src, ns_tgt)

        data_dict.update({
            'ds_mat': s,
            'perm_mat': hungarian(s, ns_src, ns_tgt),
            'aff_mat': M
        })
        return data_dict


class SchurLayer(nn.Module):
    def __init__(self, iso_matrix, n, hidden_dim):
        super().__init__()
        self.iso_matrix = iso_matrix
        self.n = n
        self.hidden_dim = hidden_dim
        self.num_irreps = iso_matrix.shape[0]
        self.weight = nn.Parameter(torch.empty(self.num_irreps, self.num_irreps))
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        # features on first dim
        x = x.transpose(1, 3)
        A, B = torch.unbind(x)
        # get all irrep projections
        a_, b_ = [], []
        for a, b in zip(A, B):
            a, b = pair_decomposition(a, b)
            a, b = torch.stack(a), torch.stack(b)
            a_.append(a)
            b_.append(b)
        a, b = torch.stack(a_), torch.stack(b_)
        # get the isomorphism scalars wcich are not 0
        params = self.weight * self.iso_matrix
        # apply schur and calculate next layer irreps
        a = (params @ a.transpose(0, 1).reshape(self.num_irreps, -1)).reshape(self.num_irreps, self.hidden_dim, self.n,
                                                                              self.n)
        b = (params @ b.transpose(0, 1).reshape(self.num_irreps, -1)).reshape(self.num_irreps, self.hidden_dim, self.n,
                                                                              self.n)
        # reconstruct
        A, B = a.sum(dim=0), b.sum(dim=0)
        return torch.stack([A, B]).transpose(1, 3)


class LinearGMLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1, bias=False)

    def forward(self, x):
        A, B = torch.unbind(x)
        a_vec = torch.stack(get_zero_sum_vectors(A.squeeze()))
        b_vec = torch.stack(get_zero_sum_vectors(B.squeeze()))
        a_vec = self.linear(a_vec.T).squeeze()
        b_vec = self.linear(b_vec.T).squeeze()
        pred = torch.outer(a_vec, b_vec)
        pred = sinkhorn(pred)
        return pred


class GraphMatchingModel(L.LightningModule):
    def __init__(self, n, feature_dim):
        super().__init__()
        self.n = n
        self.f_dim = feature_dim
        iso_matrix = get_iso_matrix()
        self.model = nn.Sequential(SchurLayer(iso_matrix, n, feature_dim),
                                   nn.Linear(feature_dim, feature_dim),
                                   nn.ReLU(),
                                   SchurLayer(iso_matrix, n, feature_dim),
                                   nn.Linear(feature_dim, 2 * feature_dim),
                                   nn.ReLU(),
                                   nn.Linear(2 * feature_dim, 1),
                                   LinearGMLayer()
                                   )

    def forward(self, A, B):
        return self.model(torch.stack([A, B]).transpose(1, 3))

    def training_step(self, batch, batch_idx):
        A, B, P = batch
        A = A.squeeze()
        B = B.squeeze()
        pred = self.model(torch.stack([A, B]).transpose(1, 3)).unsqueeze(0)
        loss = permutation_loss(pred, P)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        A, B, P = batch
        A = A.squeeze()
        B = B.squeeze()
        pred = self.model(torch.stack([A, B]).transpose(1, 3)).unsqueeze(0)
        loss = permutation_loss(pred, P)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer
