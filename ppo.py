import os

from stable_baselines3 import PPO

from env import get_atari_env

BATCH_SIZE = 256
CLIP_RANGE = 0.1
ENT_COEF = 0.01
LEARNING_RATE = 5e-4
N_EPOCHS = 4
N_STEPS = 128
POLICY = "CnnPolicy"
VF_COEF = 0.5
NORMALIZE_ADVANTAGE = False
ENV = get_atari_env()
TENSORBOARD_LOG = "BreakoutNoFrameskip-v4"
SEED = 42
DEVICE = "cuda"
MODEL_PATH = "./models/best_model.zip"


def get_model_ppo():
    if os.path.exists(MODEL_PATH):
        # load existing model
        model = PPO.load(MODEL_PATH, env=ENV, device=DEVICE)
        print("Model loaded successfully.")
    else:
        # create new model if no model exists
        model = PPO(
            policy=POLICY,
            env=ENV,
            learning_rate=LEARNING_RATE,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            clip_range=CLIP_RANGE,
            ent_coef=ENT_COEF,
            normalize_advantage=NORMALIZE_ADVANTAGE,
            vf_coef=VF_COEF,
            tensorboard_log=TENSORBOARD_LOG,
            seed=SEED,
            device=DEVICE,
        )
    return model
