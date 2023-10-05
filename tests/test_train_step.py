import jax
import jax.numpy as jnp
from symbiotic_transformers.cuisine_school import chef, chef_config, recipe_reader


def test_true():
    assert True


def test_promoted_chef_can_learn():
    config = chef_config.ChefConfig()
    brain_assistant = chef.BrainAssistant(config)
    train_ds, exam_ds = recipe_reader.ImitateChefDataset(
        brain_assistant
    ).get_recipe_datasets("./recipes/recipes_small.txt")
    promoted_chef = chef.PromotedChef(config)
    promoted_chef.become_a_master_chef(train_ds, exam_ds)
    assert True
