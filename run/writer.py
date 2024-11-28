import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

def create_writer(
    name: str = None,
    upswing: bool = False,
    target: bool = False,
    extended_observation: bool = False,
    policy_model: str = None,
    value_model: str = None,
    num_observations: int = None,
    num_epochs: int = None,
    num_runner_steps: int = None,
    gamma: float = None,
    lambda_: float = None,
    num_minibatches: int = None,
    batch_size: int = None,
    estimation_mass_model: str = None,
    epochs: int = None,
) -> SummaryWriter:
    """
    Create a SummaryWriter object for logging the training and test results.

    Args:
        Hyperparameters
        
    Returns:
        SummaryWriter: The SummaryWriter object.
    """

    timestamp = str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))

    log_dir = os.path.join(
        "runs",
        name,
        timestamp
    ).replace("\\", "/")

    comment = ', '.join([
            f"upswing={upswing}",
            f"target={target}",
            f"extended_observation={extended_observation}",
            f"policy_model={policy_model}",
            f"value_model={value_model}",
            f"num_observations={num_observations}",
            f"num_epochs={num_epochs}",
            f"num_runner_steps={num_runner_steps}",
            f"gamma={gamma}",
            f"lambda_={lambda_}",
            f"num_minibatches={num_minibatches}",
            f"batch_size={batch_size}",
            f"estimation_mass_model={estimation_mass_model}",
            f"epochs={epochs}",
        ]
    )

    writer = SummaryWriter(log_dir=log_dir)
    with open(os.path.join(log_dir, "hyperparameters.txt"), "w") as f:
        f.write(comment)
    f.close()
    return writer
