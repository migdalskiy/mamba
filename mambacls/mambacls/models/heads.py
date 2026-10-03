from typing import Optional

import torch.nn as nn


class ClassificationHead(nn.Module):
    """Dropout -> [Linear -> GELU -> Dropout] -> Linear. Only new parameters are initialised here;
    the backbone is never re-initialised (spec §2.2: dt bias init caveat)."""

    def __init__(self, d_in: int, n_classes: int, hidden: Optional[int] = None, dropout: float = 0.1,
                 multilabel: bool = False):
        super().__init__()
        self.multilabel = multilabel
        layers = [nn.Dropout(dropout)]
        if hidden:
            layers += [nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout)]
            d_in = hidden
        layers.append(nn.Linear(d_in, n_classes))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
