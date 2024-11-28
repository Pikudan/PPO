import torch


class PPO:
    """ PPO algorithm. """
    def __init__(
        self,
        policy=None,
        optimizer=None,
        cliprange=0.2,
        value_loss_coef=0.25,
        max_grad_norm=0.5
    ):
        self.policy = policy
        self.optimizer = optimizer
        self.cliprange = cliprange
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.advantages_np = None
        self.policy_loss_np = None
        self.value_loss_np = None
        self.ppo_loss_np = None
        self.total_norm = None
        self.values_np = None
        self.values_loss_np = None

    def policy_loss(self, trajectory, act):
        """ Computes and returns policy loss on a given trajectory. """
        log_probs_all = act['distribution'].log_prob(torch.tensor(trajectory['actions']))
        log_old_probs_all = torch.tensor(trajectory['log_probs'])
        ratio = (log_probs_all - log_old_probs_all).exp()
        J_pi = ratio*trajectory['advantages'].detach()
        self.advantages_np = trajectory['advantages'].detach().mean().numpy()
        J_pi_clipped = torch.clamp(ratio, 1 - self.cliprange, 1 + self.cliprange)*trajectory['advantages'].detach()
        return -torch.mean(torch.min(J_pi, J_pi_clipped))

    def value_loss(self, trajectory, act):
        """ Computes and returns value loss on a given trajectory. """
        self.values_np = trajectory['values'].detach().mean().cpu().numpy()
        L_simple = (act['values'] - trajectory['value_targets'].detach())**2
        L_clipped = (trajectory['values'] + torch.clamp(act['values'] - trajectory['values'],
                    -self.cliprange, self.cliprange) - trajectory['value_targets'].detach())**2
        self.values_loss_np = torch.mean(torch.max(L_simple, L_clipped)).detach().numpy()
        return torch.mean(torch.max(L_simple, L_clipped))

    def loss(self, trajectory):
        act = self.policy.act(trajectory["observations"], training=True)
        policy_loss = self.policy_loss(trajectory, act)
        value_loss = self.value_loss(trajectory, act)
        self.policy_loss_np = policy_loss.detach().numpy()
        self.value_loss_np = value_loss.detach().numpy()
        self.ppo_loss_np = self.policy_loss_np + self.value_loss_coef * self.value_loss_np
        return policy_loss + self.value_loss_coef * value_loss

    def step(self, trajectory):
        """ Computes the loss function and performs a single gradient step. """
        self.optimizer.zero_grad()
        self.loss(trajectory).backward()
        torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.total_norm = 0
        for p in self.policy.model.parameters():
            param_norm = p.grad.data.norm(2)
            self.total_norm += param_norm.item() ** 2
        self.total_norm = self.total_norm ** (1. / 2)
