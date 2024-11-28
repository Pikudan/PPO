from test import test
from train import train
from arguments import get_args


def main(args=None):
    if args.mode == "train":
        train(
            upswing=args.upswing,
            target=args.target,
            extended_observation=args.extended_observation,
            mass_use=args.mass_use,
            policy_model=args.policy_model,
            value_model=args.value_model,
            num_observations=args.num_observations,
            num_epochs=args.num_epochs,
            num_runner_steps=args.num_runner_steps,
            gamma=args.gamma,
            lambda_=args.lambda_,
            num_minibatches=args.num_minibatches
        )
    if args.mode == "test":
        test(
            upswing=args.upswing,
            target=args.target,
            extended_observation=args.extended_observation,
            mass_use=args.mass_use,
            mass=args.mass,
            num_observations=args.num_observations,
            policy_model=args.policy_model,
            value_model=args.value_model
        )

if __name__ == "__main__":
    args = get_args()
    main(args)
