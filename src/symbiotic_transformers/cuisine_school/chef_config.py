import os
import jax


class ChefConfig:
    def __init__(
        self,
        max_seq_len=128,
        brain_size=256,
        n_ideas=2,
        n_moldings=2,
        dropout_rate=0.2,
        num_subordinates_per_shop=32,
        n_shops=1,
        n_times_to_imitate_chefs=100,
        n_training_recipes_per_imitation=5000,
        n_exam_recipes_per_imitation=500,
        kitchen_seed=42,
        log_dir="./logs/",
        vocab_dir="/tmp/ckpt/orbax/vocabulary",
        chef_state_path="/tmp/ckpt/orbax/managed/",
        init_lr=1e-4,
        peak_lr=1e-3,
        final_lr=3e-4,
        n_warmup_steps=50000,
        n_decay_steps=500000,
        overfit_on_small_dataset=False,
        parallel_training=False,
    ):
        self.max_seq_len = max_seq_len
        self.brain_size = brain_size
        self.n_ideas = n_ideas
        self.n_moldings = n_moldings
        self.dropout_rate = dropout_rate
        self.num_subordinates_per_shop = num_subordinates_per_shop
        self.n_shops = n_shops

        self.n_times_to_imitate_chefs = n_times_to_imitate_chefs
        self.n_training_recipes_per_imitation = n_training_recipes_per_imitation
        self.n_exam_recipes_per_imitation = n_exam_recipes_per_imitation

        self.kitchen_seed = kitchen_seed

        self.log_dir = log_dir
        self.vocab_dir = vocab_dir
        self.chef_state_path = chef_state_path

        self.init_lr = init_lr
        self.peak_lr = peak_lr
        self.final_lr = final_lr
        self.n_warmup_steps = n_warmup_steps
        self.n_decay_steps = n_decay_steps
        self.overfit_on_small_dataset = overfit_on_small_dataset
        self.parallel_training = parallel_training
        if self.parallel_training:
            os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
            print("jax devices: ", jax.devices())

    # computed method as a property
    @property
    def batch_size(self):
        return (
            self.num_subordinates_per_shop * self.n_shops
            if self.parallel_training
            else self.num_subordinates_per_shop
        )

    @property
    def overfit_size_by_reducing_text(self):
        return 2000 if self.overfit_on_small_dataset else None

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
