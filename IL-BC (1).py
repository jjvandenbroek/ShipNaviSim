from dataclasses import dataclass, asdict
import pyrallis
import gymnasium as gym
import pygame
import wandb
import pandas as pd
from gymnasium import spaces
from datetime import timedelta, datetime
import numpy as np
import polars as pl
from typing import List
import utils
import minari
from imitation.data import rollout, types
from det_bc import evaluate_policy_with_finalinfo
from imitation.util import logger as imit_logger
from typing import (
    Type
)
from imitation.algorithms import bc
from stable_baselines3.common import policies, torch_layers
from torch import nn
import torch as th
from imitation.util import networks, util    
from stable_baselines3.common.vec_env import DummyVecEnv
import det_bc
import polars as pl
from ship_env import ShipEnvironment, read_position_data, chunk_to_traj
from datetime import timedelta
import utils
import pandas as pd
import numpy as np
import gymnasium as gym
import minari
import os
import csv

class CombinedNormExtractor(torch_layers.CombinedExtractor):
    """
    Combined features extractor for Dict observation spaces.
    Flatten obs of each key then concat and normalize
    """
    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 256,
        normalized_image: bool = False,
        normalize_class: Type[nn.Module] = networks.RunningNorm,
    ) -> None:
        super().__init__(observation_space, cnn_output_dim, normalized_image)
        self.normalize = normalize_class(self.features_dim)  # type: ignore[call-arg]
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        flattened = super().forward(observations)
        return self.normalize(flattened)


class EpochCSVLogger:
    """
    Logger that records training statistics to CSV after each epoch.
    Saves incrementally so progress isn't lost if training is interrupted.
    """
    def __init__(self, log_dir: str, run_name: str):
        """
        Args:
            log_dir: Directory to save CSV files
            run_name: Name for this training run (used in filename)
        """
        self.log_dir = log_dir
        self.run_name = run_name
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = os.path.join(log_dir, f"{run_name}_epoch_stats_{timestamp}.csv")
        
        # Initialize CSV file with headers
        self.fieldnames = ['epoch', 'timestamp', 'bc/loss', 'bc/l2_norm', 'bc/l2_loss', 
                          'bc/prob_true_act', 'bc/ent', 'bc/neglogp',
                          'val/loss', 'val/l2_norm', 'val/l2_loss',
                          'val/prob_true_act', 'val/ent', 'val/neglogp']
        
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
        
        print(f"Epoch statistics will be logged to: {self.csv_file}")
        
        # Track current epoch stats
        self.current_epoch = 0
        self.epoch_stats = {}
    
    def log_stat(self, key: str, value: float):
        """Record a statistic for the current epoch"""
        self.epoch_stats[key] = value
    
    def on_epoch_end(self, bc_trainer):
        """
        Called at the end of each epoch to save statistics to CSV.
        
        Args:
            bc_trainer: The BC trainer object (to access logger stats if needed)
        """
        self.current_epoch += 1
        
        # Get statistics from the BC trainer's logger
        # The logger accumulates stats during training
        row = {
            'epoch': self.current_epoch,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Add any accumulated stats
        row.update(self.epoch_stats)
        
        # Ensure all fieldnames are present (fill missing with None)
        for field in self.fieldnames:
            if field not in row:
                row[field] = None
        
        # Append to CSV file immediately (incremental save)
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)
        
        # Print progress
        print(f"Epoch {self.current_epoch} completed - stats saved to CSV")
        
        # Clear epoch stats for next epoch
        self.epoch_stats = {}
    
    def get_csv_path(self):
        """Return the path to the CSV file"""
        return self.csv_file

@dataclass
class TrainConfig:
    #Mode (train or eval or train-eval for both)
    mode: str = "train"
    # wandb project name
    project: str = "Maritime"
    # wandb group name
    group: str = "BC"
    # wandb run name
    name: str = "BC-deterministic-256-128hid-256batch-combMLPTanh"
    # training dataset name
    env: str = "Maritime-Expert-Bolivar-Roads-MINI-v2"
    #use CombinedMLP
    use_combMLP: bool = True
    #use_deterministic policy or not:
    use_det: bool = True
    # total gradient updates during training
    max_timesteps: int = int(1e6)
    #NN hid size
    hid_size = 256
    # training batch size
    batch_size: int = 256
    #Using HER 
    use_her: bool = False
    # training random seed
    seed: int = 42
    # training device (default tries cuda, but script will fallback to cpu if unavailable)
    device: str = "cuda"
    # wandb or not
    wandb: bool = False
    #checkpoint path
    ckpoint_folder: str = "./ckpoints"
    #checkpoint path
    ckpoint_path:str = "./ckpoints/BC-deterministic-256-128hid-256batch-combMLPTanh-Maritime-Expert-Bolivar-Roads-MINI-v1.th"

def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"] +"-"+config["env"],
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=False,  # optional        
    )
    wandb.run.save()

        
@pyrallis.wrap()        
def load_ships_and_play(cfg: TrainConfig):
    """
    Main training/evaluation function.
    
    SIMPLIFIED APPROACH:
    - Data is already preprocessed and stored in the Minari dataset
    - No need to re-filter, re-interpolate, or re-chunk raw data
    - For evaluation, we use a dummy environment (only spaces needed for policy)
    """
    
    # Ensure device setting is compatible with installed PyTorch
    try:
        import torch as th
        # If user requested cuda but torch wasn't built with cuda, fall back to cpu
        if cfg.device.startswith("cuda") and not th.cuda.is_available():
            print("CUDA not available - falling back to CPU for tensors and model.")
            cfg.device = "cpu"
    except Exception:
        # If torch import fails for any reason, set to cpu (safe fallback)
        cfg.device = "cpu"

    # Load the pre-built Minari dataset (contains all preprocessed trajectories)
    minari_dataset = minari.load_dataset(cfg.env)
    
    # Try to recover environment from dataset, but if it fails (no env_spec),
    # create a minimal environment using the dataset's observation/action spaces
    try:
        env = minari_dataset.recover_environment()
        print("Environment recovered from dataset")
    except (ValueError, AttributeError) as e:
        print(f"Cannot recover environment from dataset: {e}")
        print("Creating minimal environment from dataset spaces...")
        
        # Get spaces from the first episode
        first_episode = next(minari_dataset.iterate_episodes())
        
        # Create a minimal environment wrapper with just the spaces
        # This is sufficient for training (we don't roll out, just use stored data)
        class MinimalEnv(gym.Env):
            """Minimal environment that just provides observation/action spaces"""
            def __init__(self, obs_space, act_space):
                super().__init__()
                self.observation_space = obs_space
                self.action_space = act_space
                
            def reset(self, **kwargs):
                # Not used during training (data comes from dataset)
                return self.observation_space.sample(), {}
            
            def step(self, action):
                # Not used during training (data comes from dataset)
                obs = self.observation_space.sample()
                return obs, 0.0, False, False, {}
        
        # Infer spaces from dataset
        obs_space = minari_dataset.observation_space
        act_space = minari_dataset.action_space
        
        env = MinimalEnv(obs_space, act_space)
        print(f"Created minimal environment with observation and action spaces from dataset")
        # Uncomment for detailed space info:
        # print(f"  obs_space: {obs_space}")
        # print(f"  act_space: {act_space}")
    
    #BC Training
    if cfg.mode == "train" or cfg.mode == "train-eval":
        if cfg.wandb:
            wandb_init(asdict(cfg))
        
        rng = np.random.default_rng(42)
        
        # Convert Minari dataset episodes to imitation learning trajectories
        # The dataset already contains properly filtered and processed data
        trajs: List[types.TrajectoryWithRew] = []
        episode_ids = []  # Track episode indices for later split identification
        count_nb = 0
        
        print("Loading episodes from Minari dataset...")
        for episode_idx, episode_data in enumerate(minari_dataset.iterate_episodes()):
            observations = episode_data.observations                       
            actions = episode_data.actions
            rewards = episode_data.rewards
            terminations = episode_data.terminations
            truncations = episode_data.truncations
            infos = episode_data.infos
            
            # Wrap observations in proper format for imitation learning
            new_obs = types.maybe_wrap_in_dictobs(observations)
            trajs.append(types.TrajectoryWithRew(new_obs, actions, None, True, rewards))
            episode_ids.append(episode_idx)  # Store episode index
            
            # Optional: Hindsight Experience Replay (HER)
            # Creates additional training data by using reached states as goals
            if cfg.use_her:
                her_observations = observations.copy()
                for i in range(len(her_observations['ego']) - 1):
                    # Set goal to next state's position (hindsight relabeling)
                    her_observations['goal'][i] = her_observations['ego'][i+1][-1][:2]
                new_obs = types.maybe_wrap_in_dictobs(her_observations)
                trajs.append(types.TrajectoryWithRew(new_obs, actions, None, True, rewards))
            
            # Track long episodes (> 20 minutes at 10s timesteps = 120 steps)
            if(len(actions) > 120):
                count_nb += 1
            
            # Sanity check: observations should be actions + 1 (initial obs + transitions)
            assert len(new_obs) == len(actions) + 1
            assert len(rewards) == len(actions)
        
        # Create paired list of trajectories and their episode IDs before shuffling
        traj_episode_pairs = list(zip(trajs, episode_ids))
        
        # Shuffle trajectories for better training (shuffle the pairs to keep mapping)
        rng.shuffle(traj_episode_pairs)
        
        # Unzip back into separate lists
        trajs, episode_ids = zip(*traj_episode_pairs)
        trajs = list(trajs)
        episode_ids = list(episode_ids)
        
        # Split into train and test sets (80/20 split)
        train_size = int(0.8 * len(trajs))
        train_trajs = trajs[:train_size]
        test_trajs = trajs[train_size:]
        train_episode_ids = episode_ids[:train_size]
        test_episode_ids = episode_ids[train_size:]
        
        # Save episode IDs to disk for traceability
        split_info_dir = "training_stats"
        os.makedirs(split_info_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        split_file = os.path.join(split_info_dir, f"{cfg.name}_{cfg.env}_split_{timestamp}.json")
        
        split_info = {
            'timestamp': timestamp,
            'dataset_name': cfg.env,
            'run_name': cfg.name,
            'total_episodes': len(trajs),
            'train_size': len(train_trajs),
            'test_size': len(test_trajs),
            'train_episode_ids': train_episode_ids,
            'test_episode_ids': test_episode_ids,
            'random_seed': 42
        }
        
        import json
        with open(split_file, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Dataset Split:")
        print(f"  Total trajectories: {len(trajs)}")
        print(f"  Training set: {len(train_trajs)} trajectories ({len(train_trajs)/len(trajs)*100:.1f}%)")
        print(f"  Test set: {len(test_trajs)} trajectories ({len(test_trajs)/len(trajs)*100:.1f}%)")
        print(f"  Episodes longer than 20 minutes: {count_nb}")
        print(f"  Split info saved to: {split_file}")
        print(f"{'='*60}\n")
        
        # Print statistics for both sets
        train_stats = rollout.rollout_stats(train_trajs)
        test_stats = rollout.rollout_stats(test_trajs)
        print("Training set statistics:")
        print(train_stats)
        print("\nTest set statistics:")
        print(test_stats)
        
        #Training part - use only training trajectories
        transitions = rollout.flatten_trajectories(train_trajs)
        test_transitions = rollout.flatten_trajectories(test_trajs)    
        log_dir = "logs/BC"
        custom_logger = imit_logger.configure(
            folder=log_dir,
            format_strs=["tensorboard", "stdout"],
        )
        
        # Create CSV logger for epoch statistics
        training_stats_dir = "training_stats"
        csv_logger = EpochCSVLogger(training_stats_dir, f"{cfg.name}_{cfg.env}")
        
        # Create a callback to log batch statistics and save epoch stats
        batch_stats = {'total_loss': [], 'l2_norm': [], 'l2_loss': []}
        
        def on_batch_end_callback():
            """Called after each training batch"""
            # Try to get current batch loss from the logger
            # This will be accumulated for the epoch
            pass
        
        def compute_validation_loss(policy, test_transitions):
            """Compute validation loss on test set"""
            # Get test data
            test_obs = test_transitions.obs
            test_acts = test_transitions.acts
            
            # Convert to tensors using the same method as training
            with th.no_grad():
                test_obs_tensor = types.map_maybe_dict(
                    lambda x: util.safe_to_tensor(x, device=cfg.device),
                    types.maybe_unwrap_dictobs(test_obs),
                )
                test_acts_tensor = util.safe_to_tensor(test_acts, device=cfg.device)
                
                # Get policy predictions (forward pass)
                pred_acts = policy(test_obs_tensor)
                
                # Compute metrics (same as training)
                # L2 loss (MSE)
                l2_loss = th.nn.functional.mse_loss(pred_acts, test_acts_tensor)
                
                # L2 norm of the error
                l2_norm = th.norm(pred_acts - test_acts_tensor, p=2, dim=-1).mean()
                
                # For deterministic policies, other metrics are not applicable
                # Set them to 0 for consistency with CSV format
                return {
                    'val/loss': l2_loss.item(),
                    'val/l2_norm': l2_norm.item(),
                    'val/l2_loss': l2_loss.item(),
                    'val/prob_true_act': 0.0,
                    'val/ent': 0.0,
                    'val/neglogp': 0.0,
                }
        
        def on_epoch_end_callback():
            """Called after each epoch - compute and save statistics"""
            # Compute average training stats for this epoch
            if len(batch_stats['total_loss']) > 0:
                csv_logger.log_stat('bc/loss', np.mean(batch_stats['total_loss']))
                csv_logger.log_stat('bc/l2_norm', np.mean(batch_stats['l2_norm']))
                csv_logger.log_stat('bc/l2_loss', np.mean(batch_stats['l2_loss']))
                
                # Clear batch stats for next epoch
                batch_stats['total_loss'].clear()
                batch_stats['l2_norm'].clear()
                batch_stats['l2_loss'].clear()
            
            # Compute validation metrics on test set
            val_metrics = compute_validation_loss(bc_trainer.policy, test_transitions)
            for key, value in val_metrics.items():
                csv_logger.log_stat(key, value)
            
            # Print validation stats
            print(f"  Validation - Loss: {val_metrics['val/loss']:.4f}, "
                  f"L2 Norm: {val_metrics['val/l2_norm']:.4f}")
            
            # Save to CSV
            csv_logger.on_epoch_end(bc_trainer)
        
        # custom_logger=None
        if cfg.use_det is False:      
            policy = policies.ActorCriticPolicy(
                        observation_space=env.observation_space,
                        action_space=env.action_space,
                        net_arch=[cfg.hid_size, cfg.hid_size],
                        # log_std_init = 1, #deterministic policy 
                        # Set lr_schedule to max value to force error if policy.optimizer
                        # is used by mistake (should use self.optimizer instead).
                        lr_schedule=lambda _: th.finfo(th.float32).max,
                        # features_extractor_class=torch_layers.CombinedExtractor
                        features_extractor_class=CombinedNormExtractor,
                    )    
            bc_trainer = bc.BC(
                observation_space=env.observation_space,
                action_space=env.action_space,
                demonstrations=transitions,
                rng=rng,
                policy=policy,
                l2_weight=0.001,
                # ent_weight=0,
                batch_size=cfg.batch_size,
                custom_logger=custom_logger
                )        
        else:
            if cfg.use_combMLP is False:
                policy = det_bc.DetPolicy(
                            observation_space=env.observation_space,
                            action_space=env.action_space,
                            net_arch=[cfg.hid_size, cfg.hid_size],                    
                            features_extractor_class=CombinedNormExtractor,
                        )       
            else:
                policy = det_bc.DetPolicy(
                            observation_space=env.observation_space,
                            action_space=env.action_space,
                            net_arch=[cfg.hid_size],                    
                            features_extractor_class=det_bc.NewCombinedNormExtractor,
                        )            
            bc_trainer = det_bc.BC(
                observation_space=env.observation_space,
                action_space=env.action_space,
                demonstrations=transitions,
                rng=rng,
                policy=policy,
                l2_weight=0.001,            
                batch_size=cfg.batch_size,
                custom_logger=custom_logger
                )
        
        # Add custom batch tracking to accumulate stats for CSV logging
        # Hook into the loss_calculator to capture training metrics
        original_loss_calculator = bc_trainer.loss_calculator
        
        def tracked_loss_calculator(policy, obs, acts):
            """Wrapper to track training statistics"""
            training_metrics = original_loss_calculator(policy, obs, acts)
            
            # Accumulate batch statistics from BCTrainingMetrics
            batch_stats['total_loss'].append(training_metrics.loss.item())
            batch_stats['l2_norm'].append(training_metrics.l2_norm.item())
            batch_stats['l2_loss'].append(training_metrics.l2_loss.item())
            
            return training_metrics
        
        # Replace the loss calculator with our tracked version
        bc_trainer.loss_calculator = tracked_loss_calculator
        
        # Train with epoch callback
        print(f"\n{'='*60}")
        print(f"Starting training for {300} epochs...")
        print(f"Epoch statistics will be saved to: {csv_logger.get_csv_path()}")
        print(f"{'='*60}\n")
        
        bc_trainer.train(
            n_epochs=300,
            on_epoch_end=on_epoch_end_callback,
            on_batch_end=on_batch_end_callback,
            log_interval=500
        )  
        utils.create_directory(cfg.ckpoint_folder)
        util.save_policy(bc_trainer.policy, f"{cfg.ckpoint_folder}/{cfg.name}-{cfg.env}.th")
    
    # Evaluation mode - test trained policy on episodes from dataset
    if "eval" in cfg.mode:
        print("\n" + "="*60)
        print("EVALUATION MODE")
        print("="*60)
        print("\nWARNING: Full environment evaluation requires reconstructing the environment.")
        print("Since the dataset doesn't store env_spec, you have two options:")
        print("1. Evaluate using rollouts from the dataset (compare policy actions to expert)")
        print("2. Re-create the environment manually in your evaluation script")
        print("\nFor now, showing dataset-based evaluation...")
        
        # Load trained policy
        if cfg.use_det is False:
            eval_policy = bc.reconstruct_policy(f"{cfg.ckpoint_path}")
        else:
            eval_policy = det_bc.reconstruct_policy(f"{cfg.ckpoint_path}")
        
        # Evaluate by comparing policy predictions to expert actions in dataset
        lst_infos = []
        total_action_error = []
        
        print(f"\nEvaluating on {minari_dataset.total_episodes} episodes...")
        
        for ep_idx, episode_data in enumerate(minari_dataset.iterate_episodes()):
            observations = episode_data.observations
            expert_actions = episode_data.actions
            
            # Get policy predictions for each observation
            policy_actions = []
            obs_list = types.maybe_wrap_in_dictobs(observations)
            
            # Predict actions for all observations except the last one
            for i in range(len(obs_list) - 1):
                obs_dict = {key: obs_list[key][i:i+1] for key in obs_list.keys()}
                action, _ = eval_policy.predict(obs_dict, deterministic=True)
                policy_actions.append(action[0])
            
            policy_actions = np.array(policy_actions)
            
            # Compute action-level metrics
            action_errors = np.abs(policy_actions - expert_actions)
            mean_action_error = np.mean(action_errors, axis=0)
            
            # Store episode results
            episode_info = {
                'episode_id': ep_idx,
                'num_steps': len(expert_actions),
                'mean_dx_error': mean_action_error[0],
                'mean_dy_error': mean_action_error[1],
                'mean_dheading_error': mean_action_error[2],
                'total_action_error': np.mean(action_errors)
            }
            
            lst_infos.append(episode_info)
            total_action_error.append(np.mean(action_errors))
            
            if (ep_idx + 1) % 10 == 0:
                print(f"Evaluated {ep_idx + 1}/{minari_dataset.total_episodes} episodes...")
        
        # Aggregate and save evaluation results
        df_bc_infos = pd.DataFrame(lst_infos)
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS (Action-Level Comparison)")
        print("="*60)
        print(f"\nOverall Mean Action Error: {np.mean(total_action_error):.6f}")
        print("\nPer-Episode Statistics:")
        print(df_bc_infos.describe())
        
        # Save results to CSV
        results_filename = f"{cfg.name}_{cfg.env}_eval_stats.csv"
        df_bc_infos.to_csv(results_filename, sep=";", index=False)
        print(f"\nResults saved to: {results_filename}")
        
        print("\n" + "="*60)
        print("NOTE: For full environment-based evaluation with metrics like")
        print("gc_ade, mae_steer, etc., you need to:")
        print("1. Manually create the ShipEnvironment")
        print("2. Load trajectory data from the dataset or raw files")
        print("3. Run rollouts in the environment")
        print("="*60)
    
    if hasattr(env, 'close'):
        env.close()
    

if __name__ == "__main__":
    th.manual_seed(42)
    load_ships_and_play()