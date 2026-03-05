from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gdrl.env.privileged_env import GDPrivilegedEnv


def main():
    env = DummyVecEnv([lambda: GDPrivilegedEnv()])
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        device="cpu",
    )
    model.learn(total_timesteps=200_000)
    model.save("artifacts/teacher_mock")


if __name__ == "__main__":
    main()
