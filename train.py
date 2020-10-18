from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from env import Support_v0
from policy import MyActorCriticPolicy
from model import ResFeatureExtractor


def linear_schedule(initial_value, final_value):
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        return progress * (initial_value-final_value) + final_value

    return func


if __name__ == "__main__":
    seed = 0
    n_envs = 32
    features_extractor_kwargs = dict(n_channel=64, n_block=10)
    optimizer_kwargs = dict(weight_decay=1e-4)
    ppo_kwargs = dict(
        learning_rate=5e-5,
        n_steps=32,
        batch_size=64,
        n_epochs=4,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.01,
        verbose=1,
        seed=seed
    )
    policy_kwargs = dict(
        valuehead_hidden=512,
        features_extractor_class=ResFeatureExtractor,
        features_extractor_kwargs=features_extractor_kwargs,
        optimizer_kwargs=optimizer_kwargs,
    )
    env_kwargs = dict(
        dataset_dir = "../size30/train/",
        board_size = 30
    )

    env = make_vec_env(
        Support_v0,
        n_envs,
        seed=seed,
        env_kwargs=env_kwargs
    )

    now = datetime.now()
    date_time = now.strftime("%m.%d.%Y_%Hh%Mm%Ss")
    model = PPO(
        MyActorCriticPolicy,
        env,
        **ppo_kwargs,
        tensorboard_log=f"runs/{date_time}",
        policy_kwargs=policy_kwargs,
    )
    print(model.policy)
    model.save("./logs/rl_model_0_steps")

    checkpoint_callback = CheckpointCallback(
        save_freq=int(6e4), save_path="./logs/", name_prefix="rl_model"
    )
    model.learn(
        total_timesteps=int(3e6),
        reset_num_timesteps=False,
        callback=checkpoint_callback,
    )
    model.save("./logs/rl_model_last")
