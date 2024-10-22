from callbacks import get_eval_callback
from ppo import get_model_ppo

MODEL = get_model_ppo()
EVAL_CALLBACK = get_eval_callback()
PROGRESS_BAR = True
LOG_INTERVAL = 250
TOTAL_TIMESTEPS = 15000000


def train_model():
    """train the Model"""
    MODEL.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=EVAL_CALLBACK,
        progress_bar=PROGRESS_BAR,
        log_interval=LOG_INTERVAL,
    )


if __name__ == "__main__":
    train_model()
