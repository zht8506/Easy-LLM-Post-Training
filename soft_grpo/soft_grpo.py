# VeRL verl-main/verl/trainer/ppo/core_algos.py

## RL算法参数
# loss_mode=soft_clip
# loss_agg_mode=token-mean
# adv_estimator=grpo
# norm_adv_by_std_in_grpo=False
# # KL散度控制参数
# use_kl_in_reward=False
# kl_coef=0.0
# use_kl_loss=False
# kl_loss_type=low_var_kl # 即K3估计器
# kl_loss_coef=0.0
# # PPO裁剪比例参数
# clip_ratio_low=0.2
# clip_ratio_high=0.28
# clip_ratio_c=3.0
# soft_clip_alpha=80.0


@register_policy_loss("soft_clip")  # type: ignore[arg-type]
def compute_policy_loss_soft_clip(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute a soft-clipped policy objective for PPO.

    Instead of hard clipping `ratio`, this objective down-weights updates when
    `ratio` moves outside the clip range, and the weight decays as the distance
    to the clip boundary grows.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get(  # Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
        "clip_ratio_c", 3.0
    )
    soft_clip_alpha = float(config.get("soft_clip_alpha", 50.0))

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )
    assert soft_clip_alpha > 0.0, f"soft_clip_alpha should be greater than 0.0, but get value: {soft_clip_alpha}."

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    lower_bound = 1 - cliprange_low
    upper_bound = 1 + cliprange_high
    # Distance is zero inside [lower_bound, upper_bound], positive outside.
    ratio_distance = torch.relu(lower_bound - ratio) + torch.relu(ratio - upper_bound)
    # distance_scale = max(float(cliprange_low), float(cliprange_high), 1e-6)
    # normalized_distance = ratio_distance / distance_scale
    soft_weights = torch.exp(-soft_clip_alpha * ratio_distance)

    pg_losses1 = -advantages * ratio * soft_weights.detach()
    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, pg_losses1)
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, pg_losses1)
    pg_clipfrac = verl_F.masked_mean((ratio_distance > 0).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
        "actor/soft_clip_weight_mean": verl_F.masked_mean(soft_weights, response_mask).detach().item(),
        "actor/soft_clip_distance_mean": verl_F.masked_mean(ratio_distance, response_mask).detach().item(),
    }
    return pg_loss, pg_metrics
