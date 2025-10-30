import torch
from torch.nn import Module

from .common import *
from .encoders import *
from .MCNet import *
from .diverse_latent_shape_prior import *

class AutoEncoder(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoder(zdim=args.latent_dim)

        self.shape_diffusion = DiffusionShape(
            net = LatentShapeNet(point_dim=3, context_dim=0, residual=args.residual),
            var_sched = VarianceSchedule1(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )

        self.diffusion = DiffusionPoint(
            net = PointwiseNet(point_dim=3, context_dim=args.latent_dim, residual=args.residual),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )
    
    def encode(self, x):
        """
        Args:
            x:  Point clouds to be encoded, (B, N, d).
        """
        z_mu, z_sigma = self.encoder(x)
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)
        return z
    
    def diffusion_sample(self, z, num_points, flexibility, truncate_std=None, ret_traj=False):
        samples = self.diffusion.sample(num_points, context=z, flexibility=flexibility, ret_traj=False)
        return samples
