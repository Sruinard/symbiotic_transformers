import argparse
from symbiotic_transformers.cuisine_school import (
    chef,
    chef_config,
    recipe_reader,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="./recipes/recipes_small.txt")

    # parse the arguments
    args = parser.parse_args()

    print("Preparing Kitchen...")
    config = chef_config.ChefConfig()
    brain_assistant = chef.BrainAssistant(config)
    train_ds, exam_ds = recipe_reader.ImitateChefDataset(
        brain_assistant
    ).get_recipe_datasets(args.text)
    student = chef.Chef(config)
    print("Kitchen is ready!")

    print("Start Training...")
    student.become_a_master_chef(train_ds, exam_ds)
    print("Done!")
