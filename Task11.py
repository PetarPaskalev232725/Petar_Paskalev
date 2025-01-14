import wandb
from wandb.integration.sb3 import WandbCallback
from ot2_gym_wrapper import OT2_wrapper
from stable_baselines3 import PPO
import os
import argparse
from clearml import Task
from typing_extensions import TypeIs
import tensorflow

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--milestones", type=list, default=[round(i * 0.05, 2) for i in range(1, 20)])

args = parser.parse_args()

task = Task.init(project_name='Mentor Group J/Group 3', # NB: Replace YourName with your own name
                    task_name='Baseline')

task.set_base_docker('deanis/2023y2b-rl:latest')

task.execute_remotely(queue_name="default")

os.environ['WANDB_API_KEY'] = 'dde7d1a68c5d76900e0e5bc636a28254a9759c13'# Add your wandb API key here

env = OT2_wrapper(args.milestones, max_steps=1000)
model = PPO('MlpPolicy', env, verbose=1)

# initialize wandb project
run = wandb.init(project="sb3_OT2",sync_tensorboard=True)

# add tensorboard logging to the model
model = PPO('MlpPolicy', env, verbose=1, 
            learning_rate=args.learning_rate, 
            batch_size=args.batch_size, 
            n_steps=args.n_steps, 
            n_epochs=args.n_epochs, 
            tensorboard_log=f"runs/{run.id}",)

# create wandb callback
wandb_callback = WandbCallback(model_save_freq=20000,
                                model_save_path=f"models/{run.id}",
                                verbose=2,
                                )

# add wandb callback to the model training
model.learn(total_timesteps=5000000, callback=wandb_callback, progress_bar=True, reset_num_timesteps=False,tb_log_name=f"runs/{run.id}")