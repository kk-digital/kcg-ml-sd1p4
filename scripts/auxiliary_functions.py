import torch


def get_torch_distribution_from_name(name: str) -> type:
    if name == "Logistic":
        def logistic_distribution(loc, scale):
            base_distribution = torch.distributions.Uniform(0, 1)
            transforms = [
                torch.distributions.transforms.SigmoidTransform().inv,
                torch.distributions.transforms.AffineTransform(loc=loc, scale=scale),
            ]
            logistic = torch.distributions.TransformedDistribution(
                base_distribution, transforms
            )
            return logistic

        return logistic_distribution
    return torch.distributions.__dict__[name]


def get_all_torch_distributions() -> tuple:
    torch_distributions_names = torch.distributions.__all__

    torch_distributions = [
        torch.distributions.__dict__[torch_distribution_name]
        for torch_distribution_name in torch_distributions_names
    ]

    return torch_distributions_names, torch_distributions


def build_noise_samplers(
        distributions: dict
) -> dict:
    noise_samplers = {
        k: lambda shape, device=None: get_torch_distribution_from_name(k)(**v)
        .sample(shape)
        .to(device)
        for k, v in distributions.items()
    }
    return noise_samplers
