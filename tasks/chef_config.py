class ChefConfig:
    max_seq_len = 128
    brain_size = 64
    n_ideas = 4  # n_heads
    n_moldings = 2  # n_layers
    dropout_rate = 0.2

    num_subordinates_per_shop = 32  # batch_size
    n_shops = 1  # distribute over multiple gpus/tpus

    n_times_to_imitate_chefs = 10
    n_recipes_to_sample = 500
    n_exam_recipes = 50

    kitchen_seed = 42

    # store the chef state
    chef_state_path = "./ckpt/orbax/managed/"

    def __init__(self, chef_vocab_size=None):
        self.chef_vocab_size = chef_vocab_size

    # computed method as a property
    @property
    def batch_size(self):
        return self.num_subordinates_per_shop * self.n_shops
