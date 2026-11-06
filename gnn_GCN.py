
import torch
from torch.nn import Module, ModuleList, LayerNorm
from torch.nn.functional import relu, dropout
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import scatter


class GCN_GNN(Module):
    def __init__(self, data, parameters):
        super().__init__()

        self.input_size = data.x.shape[-1]
        self.output_size = data.y.shape[-1]
        self.dropout = parameters.get('dropout')
        self.layers_size = parameters.get('layers_size')
        self.num_hidden = parameters.get('hidden_layers')
        self.improved = parameters.get('improved', False)

        if parameters.get('readout') == 'add':
            self.readout = global_add_pool
        elif parameters.get('readout') == 'mean':
            self.readout = global_mean_pool
        elif parameters.get('readout') == 'max':
            self.readout = global_max_pool

        # Layer GCNConv (hidden_layers + 2 layer, come negli altri modelli)
        self.convs = ModuleList()
        self.convs.append(GCNConv(self.input_size, self.layers_size, improved=self.improved))
        for _ in range(self.num_hidden):
            self.convs.append(GCNConv(self.layers_size, self.layers_size, improved=self.improved))
        self.convs.append(GCNConv(self.layers_size, self.layers_size, improved=self.improved))

        # [MODIFICA 4] una normalizzazione per ogni layer
        self.norms = ModuleList([LayerNorm(self.layers_size) for _ in range(len(self.convs))])

        # [MODIFICA 2] lin1 riceve riassunto del grafo + ultimo nodo (2 x layers_size)
        self.lin1 = Linear(2 * self.layers_size, self.layers_size)
        self.lin2 = Linear(self.layers_size, self.output_size)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if batch is None:  # un solo grafo (fuori dal DataLoader)
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # [MODIFICA 5] peso 1 per ogni arco: senza, `improved` non ha alcun effetto
        edge_weight = torch.ones(edge_index.size(1), device=x.device)

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = conv(x, edge_index, edge_weight)
            h = norm(h)                                                   # [MODIFICA 4]
            h = relu(h)
            h = dropout(h, p=self.dropout, training=self.training)
            x = h if i == 0 else x + h                                    # [MODIFICA 4] residuo

        graph = self.readout(x, batch)

        # ultimo nodo di ogni grafo del batch (indice più alto = attività più recente)
        node_idx = torch.arange(x.size(0), device=x.device)
        last_idx = scatter(node_idx, batch, dim=0, reduce='max')
        x = torch.cat([graph, x[last_idx]], dim=-1)

        x = self.lin1(x)
        x = relu(x)
        x = dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)
