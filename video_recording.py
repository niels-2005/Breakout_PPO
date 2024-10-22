from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import (VecFrameStack, VecTransposeImage,
                                              VecVideoRecorder)

from ppo import get_model_ppo

ENV_ID = "BreakoutNoFrameskip-v4"
N_ENVS = 16
MODEL = get_model_ppo()
VIDEO_LENGTH = 2000
SEED = 42
N_VEC_FRAME_STACKS = 4
PREFIX = "ppo"
VIDEO_FOLDER = "videos"


def record_video():
    """
    :param env_id: (str)
    :param model: (RL model)
    :param video_length: (int)
    :param prefix: (str)
    :param video_folder: (str)
    """
    eval_env = make_atari_env(ENV_ID, n_envs=N_ENVS, seed=SEED)
    eval_env = VecFrameStack(eval_env, n_stack=N_VEC_FRAME_STACKS)
    eval_env = VecTransposeImage(eval_env)

    eval_env = VecVideoRecorder(
        eval_env,
        video_folder=VIDEO_FOLDER,
        record_video_trigger=lambda step: step == 0,
        video_length=VIDEO_LENGTH,
        name_prefix=PREFIX,
    )

    obs = eval_env.reset()
    for _ in range(VIDEO_LENGTH):
        action, _ = MODEL.predict(obs)
        obs, _, _, _ = eval_env.step(action)

    eval_env.close()


if __name__ == "__main__":
    record_video()
