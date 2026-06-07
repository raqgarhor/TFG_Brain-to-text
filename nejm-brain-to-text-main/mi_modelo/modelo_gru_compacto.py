import torch
from torch import nn


class TemporalConvBlock(nn.Module):
    """
    Bloque temporal ligero.

    Trabaja sobre la secuencia ya proyectada y permite aprender patrones locales
    entre ventanas temporales consecutivas antes de la GRU.
    """

    def __init__(self, channels, kernel_size=5, dropout=0.15):
        super().__init__()
        padding = kernel_size // 2
        self.norm = nn.LayerNorm(channels)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.depthwise.weight, nonlinearity="linear")
        nn.init.zeros_(self.depthwise.bias)
        nn.init.normal_(self.pointwise.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, x):
        residual = x
        y = self.norm(x)
        y = y.transpose(1, 2)
        y = self.depthwise(y)
        y = self.activation(y)
        y = self.pointwise(y)
        y = y.transpose(1, 2)
        return residual + self.dropout(y)


class CompactGRUDecoder(nn.Module):
    """
    Decodificador neuronal propuesto.

    Mantiene las ideas principales del baseline:
    - adaptacion especifica por sesion,
    - activacion Softsign,
    - agrupacion temporal por ventanas,
    - salida a clases foneticas.

    La modificacion principal es anadir una proyeccion aprendida antes de la GRU:
    ventana temporal grande -> representacion compacta -> GRU reducida.
    """

    def __init__(
        self,
        neural_dim=512,
        n_days=45,
        n_classes=41,
        patch_size=14,
        patch_stride=4,
        projection_dim=512,
        projection_dropout=0.2,
        temporal_conv=False,
        temporal_conv_kernel=5,
        temporal_conv_dropout=0.15,
        input_dropout=0.2,
        gru_units=256,
        gru_layers=2,
        gru_dropout=0.3,
        recurrent_type="gru",
    ):
        super().__init__()

        self.neural_dim = neural_dim
        self.n_days = n_days
        self.n_classes = n_classes
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.projection_dim = projection_dim
        self.temporal_conv_enabled = temporal_conv
        self.gru_units = gru_units
        self.gru_layers = gru_layers
        self.recurrent_type = recurrent_type.lower()
        if self.recurrent_type not in {"gru", "lstm"}:
            raise ValueError("recurrent_type debe ser 'gru' o 'lstm'")

        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(neural_dim)) for _ in range(n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, neural_dim)) for _ in range(n_days)]
        )
        self.day_activation = nn.Softsign()
        self.day_dropout = nn.Dropout(input_dropout)

        patched_dim = neural_dim * patch_size
        self.projection = nn.Sequential(
            nn.Linear(patched_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(projection_dropout),
        )
        self.temporal_conv = (
            TemporalConvBlock(
                channels=projection_dim,
                kernel_size=temporal_conv_kernel,
                dropout=temporal_conv_dropout,
            )
            if temporal_conv
            else nn.Identity()
        )

        recurrent_cls = nn.LSTM if self.recurrent_type == "lstm" else nn.GRU
        self.gru = recurrent_cls(
            input_size=projection_dim,
            hidden_size=gru_units,
            num_layers=gru_layers,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )

        self.output = nn.Linear(gru_units, n_classes)
        self.h0 = nn.Parameter(torch.zeros(gru_layers, 1, gru_units))
        self.c0 = (
            nn.Parameter(torch.zeros(gru_layers, 1, gru_units))
            if self.recurrent_type == "lstm"
            else None
        )

        self._init_weights()

    def _init_weights(self):
        for layer in self.projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        for name, param in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        nn.init.xavier_uniform_(self.h0)
        if self.c0 is not None:
            nn.init.xavier_uniform_(self.c0)

    def _apply_day_layer(self, x, day_idx):
        day_weights = torch.stack([self.day_weights[int(i)] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[int(i)] for i in day_idx], dim=0).unsqueeze(1)
        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_activation(x)
        return self.day_dropout(x)

    def _make_patches(self, x):
        x = x.unsqueeze(1)
        x = x.permute(0, 3, 1, 2)
        x = x.unfold(3, self.patch_size, self.patch_stride)
        x = x.squeeze(2)
        x = x.permute(0, 2, 3, 1)
        return x.reshape(x.size(0), x.size(1), -1)

    def forward(self, x, day_idx, states=None, return_state=False):
        x = self._apply_day_layer(x, day_idx)
        x = self._make_patches(x)
        x = self.projection(x)
        x = self.temporal_conv(x)

        if states is None:
            h0 = self.h0.expand(self.gru_layers, x.shape[0], self.gru_units).contiguous()
            if self.recurrent_type == "lstm":
                c0 = self.c0.expand(self.gru_layers, x.shape[0], self.gru_units).contiguous()
                states = (h0, c0)
            else:
                states = h0

        output, hidden_states = self.gru(x, states)
        logits = self.output(output)

        if return_state:
            return logits, hidden_states
        return logits
