import torch
from torch.nn import Module, ModuleList, LayerNorm
from torch.nn.functional import relu, dropout
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import scatter


class GATv2_GNN(Module):
    def __init__(self, data, parameters):
        super().__init__()

        self.input_size = data.x.shape[-1]
        self.output_size = data.y.shape[-1]
        self.dropout = parameters.get('dropout')
        self.layers_size = parameters.get('layers_size')
        self.num_hidden = parameters.get('hidden_layers')
        self.num_heads = parameters.get('num_heads', 4)        # scelto da Optuna (4 se assente)
        self.concat = parameters.get('concat', False)          # scelto da Optuna (False se assente)

        if self.concat and self.layers_size % self.num_heads != 0:
            raise ValueError(f'num_heads={self.num_heads} deve dividere layers_size={self.layers_size}')

        # dimensione di uscita di OGNI testa
        head_size = self.layers_size // self.num_heads if self.concat else self.layers_size

        if parameters.get('readout') == 'add':
            self.readout = global_add_pool
        elif parameters.get('readout') == 'mean':
            self.readout = global_mean_pool
        elif parameters.get('readout') == 'max':
            self.readout = global_max_pool

        # Layer GATv2Conv (hidden_layers + 2 layer); ognuno restituisce layers_size valori per nodo
        sizes_in = [self.input_size] + [self.layers_size] * (self.num_hidden + 1)
        self.convs = ModuleList([
            GATv2Conv(in_channels=size_in, out_channels=head_size,
                      heads=self.num_heads, concat=self.concat)
            for size_in in sizes_in
        ])

        # una normalizzazione per ogni layer
        self.norms = ModuleList([LayerNorm(self.layers_size) for _ in range(len(self.convs))])

        # [MODIFICA 2] lin1 riceve il riassunto del grafo + il vettore dell'ultimo nodo (2 x layers_size)
        self.lin1 = Linear(2 * self.layers_size, self.layers_size)
        self.lin2 = Linear(self.layers_size, self.output_size)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if batch is None:  # un solo grafo (fuori dal DataLoader)
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = conv(x, edge_index)
            h = norm(h)                                                   # [MODIFICA 4] normalizzazione
            h = relu(h)
            h = dropout(h, p=self.dropout, training=self.training)
            x = h if i == 0 else x + h                                    # [MODIFICA 4] residuo

        graph = self.readout(x, batch)                                    # riassunto del grafo

        # [MODIFICA 2] ultimo nodo di ogni grafo del batch (indice più alto = attività più recente)
        node_idx = torch.arange(x.size(0), device=x.device)
        last_idx = scatter(node_idx, batch, dim=0, reduce='max')
        last = x[last_idx]

        x = torch.cat([graph, last], dim=-1)

        x = self.lin1(x)
        x = relu(x)
        x = dropout(x, p=self.dropout, training=self.training)
        out = self.lin2(x)

        return out
