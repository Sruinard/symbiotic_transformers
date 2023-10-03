class ChefConfig:
    max_seq_len = 128
    brain_size = 256
    n_ideas = 2  # n_heads
    n_moldings = 2  # n_layers
    dropout_rate = 0.2

    num_subordinates_per_shop = 32  # batch_size
    n_shops = 1  # distribute over multiple gpus/tpus

    n_times_to_imitate_chefs = 100
    n_training_recipes_per_imitation = 5000
    n_exam_recipes_per_imitation = 100
    kitchen_seed = 42

    # store the chef state
    vocab_dir = "/tmp/ckpt/orbax/vocabulary"
    chef_state_path = "/tmp/ckpt/orbax/managed/"

    # learning rate settings
    init_lr = 1e-4
    peak_lr = 1e-3
    final_lr = 3e-4
    n_warmup_steps = 50000  # approx 1 epoch
    n_decay_steps = n_warmup_steps * 10

    overfit_size_by_reducing_text = None  # 500  # int or None

    # computed method as a property
    @property
    def batch_size(self):
        return self.num_subordinates_per_shop * self.n_shops

    # from dict
    @classmethod
    def fromdict(cls, d):
        return cls(**d)

    # to dict
    def todict(self):
        return {
            "max_seq_len": self.max_seq_len,
            "brain_size": self.brain_size,
            "n_ideas": self.n_ideas,
            "n_moldings": self.n_moldings,
            "dropout_rate": self.dropout_rate,
            "num_subordinates_per_shop": self.num_subordinates_per_shop,
            "n_shops": self.n_shops,
            "n_times_to_imitate_chefs": self.n_times_to_imitate_chefs,
            "kitchen_seed": self.kitchen_seed,
            "chef_state_path": self.chef_state_path,
            "init_lr": self.init_lr,
            "peak_lr": self.peak_lr,
            "final_lr": self.final_lr,
            "n_warmup_steps": self.n_warmup_steps,
            "n_decay_steps": self.n_decay_steps,
        }
