from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

ENV_ID = "BreakoutNoFrameskip-v4"
N_ENVS = 16
N_ENVS_EVAL = 4
SEED = 42
N_VEC_FRAME_STACKS = 4


def get_atari_env():
    """Create the Atari environment frame stacking, and image transposing"""
    env = make_atari_env(ENV_ID, n_envs=N_ENVS, seed=SEED)
    env = VecFrameStack(env, n_stack=N_VEC_FRAME_STACKS)
    env = VecTransposeImage(env)
    return env


def get_eval_env():
    """Create the Evaluation Atari environment frame stacking, and image transposing"""
    eval_env = make_atari_env(ENV_ID, n_envs=N_ENVS_EVAL, seed=SEED)
    eval_env = VecFrameStack(eval_env, n_stack=N_VEC_FRAME_STACKS)
    eval_env = VecTransposeImage(eval_env)
    return eval_env
