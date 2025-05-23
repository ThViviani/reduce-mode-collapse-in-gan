from dataclasses import dataclass

@dataclass(frozen=True)
class TrainOptions:
    """Defines a train option for models"""
    
    latent_dim: int = 100
    batch_size: int = 256
    lr: float = 2e-4
    betas: tuple = (0.5, 0.999)
    n_critic_steps: int = 1
    use_gp: bool = False
    gamma_gp: float = 1.
    latent_distance_penalty: float = 1.
    diversity_penalty: float = 10.