import argparse
from tqdm import tqdm
import jax
import jax.numpy as jnp
import os
import flax
from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key
from symbiotic_transformers.cuisine_school import (
    chef,
    chef_config,
    recipe_reader,
)

if __name__ == "__main__":
    import os

    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    print(jax.devices())
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="./recipes/recipes_small.txt")

    # parse the arguments
    args = parser.parse_args()

    print("Preparing Kitchen...")
    config = chef_config.ChefConfig(parallel_training=True)
    brain_assistant = chef.BrainAssistant(config)
    train_ds, exam_ds = recipe_reader.ImitateChefDataset(
        brain_assistant
    ).get_recipe_datasets(args.text)
    chef = chef.PromotedChef(config)
    chef.become_a_master_chef(train_ds, exam_ds)
