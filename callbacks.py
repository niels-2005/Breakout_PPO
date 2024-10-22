from stable_baselines3.common.callbacks import EvalCallback

from env import get_atari_env, get_eval_env

N_ENVS = 16
EVAL_FREQ = int(1e5)
EVAL_FREQ = max(EVAL_FREQ // N_ENVS, 1)
ENV = get_atari_env()
EVAL_ENV = get_eval_env()
BEST_MODEL_SAVE_PATH = "./models/"
N_EVAL_EPISODES = 10


def get_eval_callback():
    eval_callback = EvalCallback(
        EVAL_ENV,
        best_model_save_path=BEST_MODEL_SAVE_PATH,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
    )
    return eval_callback
