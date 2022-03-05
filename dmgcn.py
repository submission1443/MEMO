import torch
import torch.nn as nn
import torch.nn.functional as F
from aggdim_dropout_v3 import DimAggregator
from aggdim_dropout_v3 import Combine


class CoupledRNN(nn.Module):
    def __init__(self, emb_size, num_user, num_loc, time_threshold, dis_threshold, device):
        super(CoupledRNN, self).__init__()
        self.device = device

        self.emb_size = emb_size
        self.num_user = num_user
        self.num_loc = num_loc

        # time threshold and dis threshold
        self.time_threshold = time_threshold
        self.dis_threshold = dis_threshold

        self.user_rnn = nn.RNNCell(3 * self.emb_size, self.emb_size).to(self.device)
        self.loc_rnn = nn.RNNCell(self.emb_size, self.emb_size).to(self.device)

        self.final_activation = nn.Tanh()

        # 1: one user, the second and the third 1 is for stop token
        self.fc = nn.Linear((1 + self.num_loc + 1) * emb_size, num_loc + 1)

        # initialize time_diff matrix, 0: short time, 1: long time
        self.time_emb = nn.Parameter(torch.randn(2, self.emb_size)).to(self.device)

        # initialize dis_diff matrix, 0: short dis, 1: long dis
        self.dis_emb = nn.Parameter(torch.randn(2, self.emb_size)).to(self.device)
        self.time_dis_act = nn.Tanh()

    def forward(self, user_emb, loc_emb, emb_matrix, time_diff, dis_diff):

        # select time and dis embedding
        time_idx = torch.LongTensor((time_diff.cpu() > self.time_threshold).long())
        selected_time = self.time_emb[time_idx]
        selected_time = selected_time**2 + self.time_dis_act(selected_time)


        dis_idx = torch.LongTensor((dis_diff.cpu() > self.dis_threshold).long())
        selected_dis = self.dis_emb[dis_idx]
        selected_dis = selected_dis ** 2 + self.time_dis_act(selected_dis)

        user_emb_time_dis = torch.cat((loc_emb, selected_time, selected_dis), dim=1)

        user_embedding_output = self.user_rnn(user_emb_time_dis, user_emb)
        loc_embedding_output = self.loc_rnn(user_emb, loc_emb)

        user_embedding_output = F.normalize(user_embedding_output)
        loc_embedding_output = F.normalize(loc_embedding_output)


        # -1 to exclude stop token
        loc_emb_all = emb_matrix[self.num_user:, :]
        pred_input = torch.cat((user_embedding_output, loc_emb_all.view(1, -1).repeat(user_embedding_output.size(0), 1)), 1)

        # out = torch.cat((user_embedding_output, loc_embedding_output), 1)
        out = self.final_activation(pred_input)
        y = self.fc(out)
        score = F.log_softmax(y, dim=1)
        return score, user_embedding_output, loc_embedding_output



class DMGCN(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, num_dims, input_size, output_sizes, emb_size, device, cuda, gcn=False):
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(DMGCN, self).__init__()
        self.device = device
        #self.features = features.to(self.device)
        self.gcn = gcn
        self.cuda = cuda
        self.dim_agg = DimAggregator(num_dims, self.cuda, self.gcn, input_size, output_sizes[0])
        self.combine2general = Combine(int(num_dims * output_sizes[0]), output_sizes[1], self.cuda)
        self.dropout = nn.Dropout(p=0)

        self.emb_size = emb_size
        self.user_rnn = nn.RNNCell(self.emb_size, self.emb_size)
        self.loc_rnn = nn.RNNCell(self.emb_size, self.emb_size)


        #self.final_activation = nn.Tanh()
        #self.h2o = nn.Linear(64, 560)
        self.linear_layers_for_dims = nn.ModuleList(
            [nn.Linear(output_sizes[1], output_sizes[1], bias=False).to(device) for i in range(num_dims)])
        self.act = nn.ELU().to(device)

    def forward(self, features, dims, counts, source_nodes, source_to_neighs_dims, target_nodes, target_to_neighs_dims,
                num_samples=10):

        x_sources = self.dim_agg(features, source_nodes, source_to_neighs_dims, num_samples)
        x_targets = self.dim_agg(features, target_nodes, target_to_neighs_dims, num_samples)

        return x_sources, x_targets

        # return outputs_user_emb, outputs_loc_emb, output
