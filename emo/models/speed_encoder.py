import torch
import torch.nn as nn
import math
from diffusers.models.modeling_utils import ModelMixin

### Code borrowed from https://github.com/johndpope/Emote-hack
class SpeedEncoder(ModelMixin):
    def __init__(self, num_speed_buckets, speed_embedding_dim):
        super().__init__()
        assert isinstance(num_speed_buckets, int), "num_speed_buckets must be an integer"
        assert num_speed_buckets > 0, "num_speed_buckets must be positive"
        assert isinstance(speed_embedding_dim, int), "speed_embedding_dim must be an integer"
        assert speed_embedding_dim > 0, "speed_embedding_dim must be positive"

        self.num_speed_buckets = num_speed_buckets
        self.speed_embedding_dim = speed_embedding_dim
        self.bucket_centers = self.get_bucket_centers()
        self.bucket_radii = self.get_bucket_radius()

        self.mlp = nn.Sequential(
            nn.Linear(num_speed_buckets**2, speed_embedding_dim//2),
            nn.ReLU(),
            nn.Linear(speed_embedding_dim//2, speed_embedding_dim)
        )

    def get_bucket_centers(self):
        """precompute speed bucket centers
        Returns:
            centers: a vector of speed bucket centers
        """
        #return [-1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0]
        centers = torch.linspace(
            -math.pi, 
            math.pi, 
            self.num_speed_buckets
        ).repeat(self.num_speed_buckets)
        return centers

    def get_bucket_radius(self):
        """precompute speed bucket radius
        Returns:
            radius: a vector of speed bucket radius
        """
        # return [0.1] * self.num_speed_buckets
        # return torch.ones(self.num_speed_buckets) * 0.1
        radius = torch.linspace(
            0.01, 
            math.pi, 
            self.num_speed_buckets
        ).repeat_interleave(self.num_speed_buckets)
        radius = 1.0 / (3.0 * radius)
        return radius

    def encode_speed(self, head_speeds):
        """
        Args:
            head_speeds: a Scalar of frame head velocity in shape (batch_size,)
        Returns:
            speed_vectors: a Tensor of speed vectors encoded with speed buckets (batch_size, num_speed_buckets**2)
        """
        speed_vectors = torch.tanh(
            (head_speeds - self.bucket_centers) * (self.bucket_radii)
        )
        return speed_vectors

    def forward(self, head_speeds):
        """
        Args:
            head_speeds
        Returns:
            speed_embeddings
        """
        if head_speeds.shape[-1] != 1:
            head_speeds = head_speeds.unsqueeze(-1)
        # Process the batch of head rotation speeds through the encoder
        speed_vectors = self.encode_speed(head_speeds)

        # Pass the encoded vectors through the MLP
        speed_embeddings = self.mlp(speed_vectors)
        return speed_embeddings