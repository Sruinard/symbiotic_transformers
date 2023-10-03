import argparse
from symbiotic_transformers.cuisine_school import (
    chef_config,
    chef_gpt_brain,
    recipe_reader,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="./recipes/recipes_small.txt")

    # parse the arguments
    args = parser.parse_args()

    print("Preparing Kitchen...")
    config = chef_config.ChefConfig()
    chef = chef_gpt_brain.Chef(config)
    training_ds, exam_ds = recipe_reader.ImitateChefDataset(
        chef=chef
    ).get_recipe_datasets(args.text)
    print("Kitchen is ready!")

    print("Start Training...")
    chef.become_a_master_chef(training_ds, exam_ds)
    print("Done!")
