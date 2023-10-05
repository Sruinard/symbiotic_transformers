import os
import datetime
from typing import Iterable

import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import flax
from flax.training import train_state, orbax_utils
from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key
import orbax
from tqdm import tqdm

from symbiotic_transformers.cuisine_school import chef_config, brain_anatomy


class BrainAssistant:
    """
    This class is responsible for translating raw inputs into features that the chef can understand.
    It can also translate the chef's output into human readable text.
    """

    def __init__(self, config: chef_config.ChefConfig):
        self.config = config
        self.stoi: tf.keras.layers.StringLookup = tf.keras.layers.StringLookup()
        self.itos: tf.keras.layers.StringLookup = tf.keras.layers.StringLookup(
            invert=True
        )

    def _clean_text_and_convert_to_characters(self, text):
        # convert to characters
        characters = tf.strings.unicode_split(text, input_encoding="UTF-8")
        # keep only letters
        characters = tf.strings.regex_replace(characters, "[^a-zA-Z ]", "")
        # to lower case
        characters = tf.strings.lower(characters)
        return characters

    def _build_chef_vocabulary(self, text: Iterable[str]):
        self.stoi = tf.keras.layers.StringLookup()
        self.stoi.adapt(text)
        self.itos = tf.keras.layers.StringLookup(
            vocabulary=self.stoi.get_vocabulary(), invert=True
        )

    def transform_recipe_for_training(self, text):
        characters = self._clean_text_and_convert_to_characters(text)
        # convert to tokens
        tokens = self.stoi(characters)
        return tokens

    def decode_chefs_articulated_idea(self, next_character_logits):
        next_character_index = jnp.argmax(
            next_character_logits, axis=-1
        )  # shape: (batch_size, T, vocab_size) --> (batch_size, T)
        return tf.strings.reduce_join(
            self.itos(next_character_index), axis=1
        )  # shape: (batch_size, T) --> (batch_size,)

    def load_vocab(self):
        if not os.path.exists(self.config.vocab_dir):
            # raise FileNotFoundError(f"vocab_dir: {self.config.vocab_dir} does not exist")
            raise FileNotFoundError(
                f"vocab_dir: {self.config.vocab_dir} does not exist"
            )
        self.stoi.load_assets(self.config.vocab_dir)
        self.itos.load_assets(self.config.vocab_dir)


class Chef:
    def __init__(self, config: chef_config.ChefConfig):
        self.config = config
        self.brain_assistant = BrainAssistant(config)
        self.checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        self.brain_assistant.load_vocab()
        self.chef_brain = self.prepare_chef_for_training(config)
        self.file_writer = tf.summary.create_file_writer(
            os.path.join(
                config.log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            )
        )

    # def init_checkpointer(self):
    #     orbax_checkpointer = checkpoint.PyTreeCheckpointer()
    #     options = checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
    #     checkpoint_manager = checkpoint.CheckpointManager(
    #         self.config.chef_state_path, orbax_checkpointer, options
    #     )
    #     return checkpoint_manager

    @property
    def checkpoint(self):
        return {
            "model": self.chef_brain,
            "config": self.config.todict(),
        }

    def load_chefs_brain(self):
        state_restored = self.checkpointer.restore(
            self.config.chef_state_path, item=self.checkpoint
        )
        self.config = chef_config.ChefConfig.fromdict(state_restored["config"])
        self.chef_brain = state_restored["model"]

    def become_a_master_chef(self, train_ds, test_ds):
        rng = jax.random.PRNGKey(self.config.kitchen_seed)

        with tqdm(total=self.config.n_times_to_imitate_chefs) as pbar:
            for replication in range(self.config.n_times_to_imitate_chefs):
                losses = {"train": [], "eval": []}
                for _ in range(self.config.n_training_recipes_per_imitation):
                    recipe_instructions, recipe_by_master = next(train_ds)
                    rng, dropout_rng = jax.random.split(rng)
                    self.chef_brain, (loss, logits) = train_step_jitted(
                        self.chef_brain,
                        recipe_instructions,
                        recipe_by_master,
                        dropout_rng,
                    )
                    losses["train"].append(float(loss))

                for _ in range(self.config.n_exam_recipes_per_imitation):
                    recipe_instructions, recipe_by_master = next(test_ds)
                    rng, dropout_rng = jax.random.split(rng)
                    loss, logits = eval_step_jitted(
                        self.chef_brain,
                        recipe_instructions,
                        recipe_by_master,
                        dropout_rng,
                    )
                    losses["eval"].append(float(loss))

                avg_train_loss = jnp.mean(jnp.array(losses["train"]))
                avg_eval_loss = jnp.mean(jnp.array(losses["eval"]))
                self.write_to_tensorboard(
                    avg_train_loss,
                    logits,
                    recipe_by_master,
                    replication,
                    scope="train",
                )
                self.write_to_tensorboard(
                    avg_eval_loss,
                    logits,
                    recipe_by_master,
                    replication,
                    scope="eval",
                )
                pbar.update(1)
                pbar.set_description(
                    f"replication: {replication} train loss: {avg_train_loss} eval loss: {avg_eval_loss}"
                )

                save_args = orbax_utils.save_args_from_target(self.checkpoint)
                self.checkpointer.save(
                    self.config.chef_state_path,
                    self.checkpoint,
                    save_args=save_args,
                    force=True,
                )

    def write_to_tensorboard(self, loss, logits, recipe_by_master, replication, scope):
        with self.file_writer.as_default():
            with tf.name_scope(scope):
                tf.summary.scalar("loss", loss, step=replication)

                with tf.name_scope("samples"):
                    tf.summary.text(
                        "predicted",
                        f"""
                                    MasterChef: {self.brain_assistant.decode_chefs_articulated_idea(recipe_by_master)[0]}
                                    StudenChef: {self.brain_assistant.decode_chefs_articulated_idea(logits)[0]}
                                    """,
                        step=0,
                    )

    def generate_new_recipe(self, title, recipe_length=500):
        template = f"title: {title} instructions: "

        state = flax.jax_utils.unreplicate(self.chef_brain)
        tokens = self.brain_assistant.transform_recipe_for_training([template])
        tokens = jnp.array(tokens.numpy())
        prepadding = max(0, self.config.max_seq_len + 1 - tokens.shape[1])
        tokens = jnp.reshape(
            jnp.pad(tokens[0], (prepadding, 0), "constant", constant_values=0), (1, -1)
        )
        for _ in range(100):
            # pylint: disable=not-callable
            character_logits = state.apply_fn(
                {"params": state.params},
                tokens[:, -self.config.max_seq_len :],
                rngs={"dropout": jax.random.PRNGKey(0)},
                training=False,
            )
            next_character_logits = character_logits[:, -1:, :]
            tokens = jnp.concatenate(
                [tokens, jnp.argmax(next_character_logits, axis=-1)], axis=-1
            )
        tokens = tokens[:, prepadding:]
        return tf.strings.reduce_join(self.brain_assistant.itos(tokens), axis=1)

    def prepare_chef_for_training(self, chef_config: chef_config.ChefConfig):
        rng = jax.random.PRNGKey(chef_config.kitchen_seed)

        model_rng, dropout_rng = jax.random.split(rng)
        # learning_rate = 3e-4
        # tx = optax.adam(learning_rate=learning_rate)

        # warmup_cosine_decay_scheduler
        learn_by_scanning_followed_by_deepdive = optax.warmup_cosine_decay_schedule(
            init_value=self.config.init_lr,
            peak_value=self.config.peak_lr,
            warmup_steps=self.config.n_warmup_steps,
            decay_steps=self.config.n_decay_steps,
            end_value=self.config.final_lr,
        )
        tx = optax.adam(learn_by_scanning_followed_by_deepdive)

        print("vocab size: ", self.brain_assistant.stoi.vocabulary_size())

        model = brain_anatomy.ChefBrain(
            max_seq_len=chef_config.max_seq_len,
            brain_size=chef_config.brain_size,
            n_ideas=chef_config.n_ideas,
            n_moldings=chef_config.n_moldings,
            dropout_rate=chef_config.dropout_rate,
            chef_vocabulary_size=self.brain_assistant.stoi.vocabulary_size(),
        )

        variables = model.init(
            {"params": model_rng, "dropout": dropout_rng},
            jnp.ones(
                (chef_config.batch_size, chef_config.max_seq_len), dtype=jnp.int32
            ),
        )

        state = train_state.TrainState.create(
            apply_fn=model.apply, tx=tx, params=variables["params"]
        )
        return state


class PromotedChef(Chef):
    """
    This class enables trying in a distributed fashion.
    Three main alterations have been added to enable this:
    1. The model is replicated across multiple devices.
    2. The train and eval steps are mapped to multiple devices.
    3. The loss is averaged across devices.
    """

    def __init__(self, config: chef_config.ChefConfig):
        super().__init__(config)
        self.chef_brain = flax.jax_utils.replicate(self.chef_brain)

    @property
    def checkpoint(self):
        return {
            "model": flax.jax_utils.unreplicate(self.chef_brain),
            "config": self.config.todict(),
        }

    def memorize_single_book(self, x, y, store_brain=True):
        ptrain_step = jax.pmap(parallel_train_step_jitted, axis_name="batch")
        rng = jax.random.PRNGKey(self.config.kitchen_seed)
        dropout_rngs = jax.random.split(rng, jax.local_device_count())

        recipe_instructions, recipe_by_master = shard(x), shard(y)

        with tqdm(total=self.config.n_training_recipes_per_imitation) as pbar:
            for _ in range(self.config.n_training_recipes_per_imitation):
                self.chef_brain, (loss, logits, metrics) = ptrain_step(
                    self.chef_brain,
                    recipe_instructions,
                    recipe_by_master,
                    dropout_rngs,
                )
                dropout_rngs = jax.random.split(
                    dropout_rngs[0], jax.local_device_count()
                )

                avg_loss = jnp.mean(loss)
                pbar.update(1)
                pbar.set_description(f"iteration: {pbar.n} train loss: {avg_loss} ")

        if store_brain:
            save_args = orbax_utils.save_args_from_target(self.checkpoint)
            self.checkpointer.save(
                self.config.chef_state_path,
                self.checkpoint,
                save_args=save_args,
                force=True,
            )

    def load_chefs_brain(self):
        super().load_chefs_brain()
        self.chef_brain = flax.jax_utils.replicate(self.chef_brain)

    def compare_sample(self, x, y):
        rng = jax.random.PRNGKey(self.config.kitchen_seed)
        dropout_rngs = jax.random.split(rng, jax.local_device_count())
        x, y = shard(x), shard(y)
        peval_step = jax.pmap(parallel_eval_step_jitted, axis_name="batch")
        loss, logits = peval_step(self.chef_brain, x, y, dropout_rngs)
        yhat_decoded, yt_decoded = self.brain_assistant.decode_chefs_articulated_idea(
            logits[0]
        ), self.brain_assistant.decode_chefs_articulated_idea(y[0])
        return yhat_decoded, yt_decoded

    def become_a_master_chef(self, train_ds, test_ds):
        ptrain_step = jax.pmap(parallel_train_step_jitted, axis_name="batch")
        peval_step = jax.pmap(parallel_eval_step_jitted, axis_name="batch")
        rng = jax.random.PRNGKey(self.config.kitchen_seed)
        dropout_rngs = jax.random.split(rng, jax.local_device_count())

        with tqdm(total=self.config.n_times_to_imitate_chefs) as pbar:
            for replication in range(self.config.n_times_to_imitate_chefs):
                losses = {"train": [], "eval": []}
                n_steps_per_train_epoch = (
                    self.config.n_training_recipes_per_imitation
                    // self.config.batch_size
                ) + 1
                n_steps_per_eval_epoch = (
                    self.config.n_exam_recipes_per_imitation // self.config.batch_size
                ) + 1
                for _ in range(n_steps_per_train_epoch):
                    recipe_instructions, recipe_by_master = next(train_ds)
                    recipe_instructions, recipe_by_master = shard(
                        recipe_instructions
                    ), shard(recipe_by_master)
                    self.chef_brain, (loss, logits, metrics) = ptrain_step(
                        self.chef_brain,
                        recipe_instructions,
                        recipe_by_master,
                        dropout_rngs,
                    )
                    dropout_rngs = jax.random.split(
                        dropout_rngs[0], jax.local_device_count()
                    )

                    avg_loss = jnp.mean(loss)
                    losses["train"].append(float(avg_loss))

                for _ in range(n_steps_per_eval_epoch):
                    recipe_instructions, recipe_by_master = next(test_ds)
                    recipe_instructions, recipe_by_master = shard(
                        recipe_instructions
                    ), shard(recipe_by_master)
                    loss, logits = peval_step(
                        self.chef_brain,
                        recipe_instructions,
                        recipe_by_master,
                        dropout_rngs,
                    )
                    avg_loss = jnp.mean(loss)
                    dropout_rngs = jax.random.split(
                        dropout_rngs[0], jax.local_device_count()
                    )
                    losses["eval"].append(float(avg_loss))

                avg_train_loss = jnp.mean(jnp.array(losses["train"]))
                avg_eval_loss = jnp.mean(jnp.array(losses["eval"]))
                self.write_to_tensorboard(
                    avg_train_loss,
                    logits[0],
                    recipe_by_master[0],
                    replication,
                    scope="train",
                )
                self.write_to_tensorboard(
                    avg_eval_loss,
                    logits[0],
                    recipe_by_master[0],
                    replication,
                    scope="eval",
                )
                pbar.update(1)
                pbar.set_description(
                    f"replication: {replication} train loss: {avg_train_loss} eval loss: {avg_eval_loss}"
                )

                save_args = orbax_utils.save_args_from_target(self.checkpoint)
                self.checkpointer.save(
                    self.config.chef_state_path,
                    self.checkpoint,
                    save_args=save_args,
                    force=True,
                )


# train and eval
def train_step(state, x, y, dropout_rng=None):
    dropout_rng = jax.random.fold_in(dropout_rng, state.step)

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, x, rngs={"dropout": dropout_rng})
        loss = jnp.mean(optax.softmax_cross_entropy(logits, y))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, (loss, logits)


@jax.jit
def train_step_jitted(state, x, y, dropout_rng=None):
    dropout_rng = jax.random.fold_in(dropout_rng, state.step)

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, x, rngs={"dropout": dropout_rng})
        loss = jnp.mean(optax.softmax_cross_entropy(logits, y))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, (loss, logits)


@jax.jit
def eval_step_jitted(state, x, y, dropout_rng=None):
    logits = state.apply_fn(
        {"params": state.params}, x, rngs={"dropout": dropout_rng}, training=False
    )
    loss = jnp.mean(optax.softmax_cross_entropy(logits, y))
    return loss, logits


@jax.jit
def parallel_train_step_jitted(state, x, y, dropout_rng=None):
    # dropout_rng = jax.random.fold_in(dropout_rng, state.step)

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, x, rngs={"dropout": dropout_rng})
        loss = jnp.mean(optax.softmax_cross_entropy(logits, y))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    state = state.apply_gradients(grads=grads)
    metrics = {"loss": jax.lax.pmean(loss, axis_name="batch")}
    return state, (loss, logits, metrics)


@jax.jit
def parallel_eval_step_jitted(state, x, y, dropout_rng=None):
    logits = state.apply_fn(
        {"params": state.params}, x, rngs={"dropout": dropout_rng}, training=False
    )
    loss = jax.lax.pmean(optax.softmax_cross_entropy(logits, y), axis_name="batch")
    return loss, logits
