"""Discriminator model for ADDA."""

from torch import nn


class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dims, hidden_dims, output_dims):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.LeakyReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.LeakyReLU(),
            nn.Linear(hidden_dims, output_dims),
            # nn.LogSoftmax()  ##二分类不用softmax，而用sigmoid
            # nn.Sigmoid()

        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out

