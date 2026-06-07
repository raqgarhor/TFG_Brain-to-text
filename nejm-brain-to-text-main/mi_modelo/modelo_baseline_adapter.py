import torch
from torch import nn

from rnn_model import GRUDecoder


class ResidualAdapter(nn.Module):
    """
    Adaptador residual ligero sobre la salida de la GRU.

    Se inicializa para que al principio apenas cambie el baseline. Durante el
    fine-tuning aprende una correccion pequena de la representacion recurrente.
    """

    def __init__(self, hidden_size=768, bottleneck=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, hidden_size),
        )

        nn.init.xavier_uniform_(self.net[1].weight)
        nn.init.zeros_(self.net[1].bias)
        nn.init.zeros_(self.net[4].weight)
        nn.init.zeros_(self.net[4].bias)

    def forward(self, x):
        return x + self.net(x)


class LogitTemporalAdapter(nn.Module):
    """
    Corrector residual sobre logits foneticos.

    Usa contexto temporal cercano para ajustar las puntuaciones de salida sin
    cambiar el espacio de clases. Se inicializa a cero para no alterar el
    baseline al comienzo del entrenamiento.
    """

    def __init__(self, n_classes=41, hidden_channels=64, kernel_size=5, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.LayerNorm(n_classes),
            nn.Linear(n_classes, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=hidden_channels,
            ),
            nn.GELU(),
            nn.Conv1d(hidden_channels, n_classes, kernel_size=1),
        )

        nn.init.xavier_uniform_(self.net[1].weight)
        nn.init.zeros_(self.net[1].bias)
        nn.init.kaiming_uniform_(self.net[4].weight, nonlinearity="linear")
        nn.init.zeros_(self.net[4].bias)
        nn.init.zeros_(self.net[6].weight)
        nn.init.zeros_(self.net[6].bias)

    def forward(self, logits):
        y = self.net[0](logits)
        y = self.net[1](y)
        y = self.net[2](y)
        y = self.net[3](y)
        y = y.transpose(1, 2)
        y = self.net[4](y)
        y = self.net[5](y)
        y = self.net[6](y)
        y = y.transpose(1, 2)
        return logits + y


class BaselineGRUWithAdapter(GRUDecoder):
    """
    Copia del baseline RNN-GRU con un adaptador residual anadido.

    Mantiene la arquitectura original hasta la salida de la GRU y anade una
    transformacion residual antes de la capa final de clases foneticas.
    """

    def __init__(
        self,
        *args,
        adapter_bottleneck=128,
        adapter_dropout=0.1,
        logit_adapter=False,
        logit_adapter_channels=64,
        logit_adapter_dropout=0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.adapter = ResidualAdapter(
            hidden_size=self.n_units,
            bottleneck=adapter_bottleneck,
            dropout=adapter_dropout,
        )
        self.logit_adapter = (
            LogitTemporalAdapter(
                n_classes=self.n_classes,
                hidden_channels=logit_adapter_channels,
                dropout=logit_adapter_dropout,
            )
            if logit_adapter
            else nn.Identity()
        )

    def forward(self, x, day_idx, states=None, return_state=False):
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)

        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        if self.patch_size > 0:
            x = x.unsqueeze(1)
            x = x.permute(0, 3, 1, 2)
            x_unfold = x.unfold(3, self.patch_size, self.patch_stride)
            x_unfold = x_unfold.squeeze(2)
            x_unfold = x_unfold.permute(0, 2, 3, 1)
            x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1)

        if states is None:
            states = self.h0.expand(self.n_layers, x.shape[0], self.n_units).contiguous()

        output, hidden_states = self.gru(x, states)
        output = self.adapter(output)
        logits = self.out(output)
        logits = self.logit_adapter(logits)

        if return_state:
            return logits, hidden_states
        return logits
