import os
from typing import Iterable

import chex
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
from flax.training import train_state
from orbax import checkpoint
from tqdm import tqdm

from symbiotic_transformers.cuisine_school import chef_config


class BrainAssistant:
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
        self.checkpointer = self.init_checkpointer()
        self.brain_assistant.load_vocab()
        self.chef_brain = self.prepare_chef_for_training(config)

    def init_checkpointer(self):
        orbax_checkpointer = checkpoint.PyTreeCheckpointer()
        options = checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
        checkpoint_manager = checkpoint.CheckpointManager(
            self.config.chef_state_path, orbax_checkpointer, options
        )
        return checkpoint_manager

    @property
    def checkpoint(self):
        return {
            "model": self.chef_brain,
            "config": self.config.todict(),
        }

    def load_chefs_brain(self):
        step = self.checkpointer.latest_step()
        state_restored = self.checkpointer.restore(step, items=self.checkpoint)
        self.config = chef_config.ChefConfig.fromdict(state_restored["config"])
        self.chef_brain = state_restored["model"]

    def become_a_master_chef(self, train_ds, test_ds):
        rng = jax.random.PRNGKey(self.config.kitchen_seed)

        for replication in range(self.config.n_times_to_imitate_chefs):
            losses = {"train": [], "eval": []}

            for batch_idx, (recipe_instructions, recipe_by_master) in tqdm(
                enumerate(train_ds)
            ):
                rng, dropout_rng = jax.random.split(rng)
                self.chef_brain, (loss, logits) = train_step_jitted(
                    self.chef_brain, recipe_instructions, recipe_by_master, dropout_rng
                )
                losses["train"].append(float(loss))

                if batch_idx > self.config.n_training_recipes_per_imitation:
                    break

            for batch_idx, (recipe_instructions, recipe_by_master) in tqdm(
                enumerate(test_ds)
            ):
                rng, dropout_rng = jax.random.split(rng)
                loss, logits = eval_step_jitted(
                    self.chef_brain, recipe_instructions, recipe_by_master, dropout_rng
                )
                losses["eval"].append(float(loss))
                if batch_idx > self.config.n_exam_recipes_per_imitation:
                    break

            print(
                f"replication: {replication} train loss: {jnp.mean(jnp.array(losses['train']))} eval loss: {jnp.mean(jnp.array(losses['eval']))}"
            )

            print(
                f"""
                recipe as written by master:
                {self.brain_assistant.decode_chefs_articulated_idea(recipe_by_master)[:3]}
                recipe as written by chef:
                {self.brain_assistant.decode_chefs_articulated_idea(logits)[:3]}
                """
            )
            self.checkpointer.save(replication, self.checkpoint)

    def generate_new_recipe(self, title, recipe_length=500):
        template = f"title: {title} instructions: "

        tokens = self.brain_assistant.transform_recipe_for_training(template)
        for _ in range(recipe_length):
            # pylint: disable=not-callable
            character_logits = self.chef_brain.apply_fn(
                {"params": self.chef_brain.params},
                jnp.asarray(tokens),
                rngs={"dropout": jax.random.PRNGKey(0)},
                training=False,
            )
            next_character_logits = character_logits[:, -1, :]
            tokens = jnp.concatenate(
                [tokens, jnp.argmax(next_character_logits, axis=-1)], axis=-1
            )
        self.brain_assistant.decode_chefs_articulated_idea(tokens)
        return

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

        model = ChefBrain(
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


# LAYERS
class StructureInformation(nn.Module):
    max_seq_len: int
    brain_size: int
    chef_vocab_size: int

    @nn.compact
    def __call__(self, x):
        # x: (batch_size, seq_len)
        # we want to add a layer norm to the features dimension
        B, T = jnp.shape(x)

        token_embeddings = nn.Embed(self.chef_vocab_size, self.brain_size)(
            x
        )  # (batch_size, seq_len, brain_size)
        position_embeddings = nn.Embed(self.max_seq_len, self.brain_size)(
            jnp.arange(0, T)
        )

        x = token_embeddings + position_embeddings
        return x


# Could have been a function, but we want to use the @nn module
class Thought(nn.Module):
    # Scaled dot product attention
    brain_size: int

    def setup(self):
        self.questions_to_improve_knowledge = nn.Dense(
            features=self.brain_size
        )  # queries
        self.knowledge_index = nn.Dense(features=self.brain_size)  # keys
        self.information_in_knowledge_index = nn.Dense(
            features=self.brain_size
        )  # values

    @nn.compact
    def __call__(
        self, x
    ):  # given_a_question, knowledge_index, information_in_knowledge_index):
        B, T, C = jnp.shape(x)

        formulated_question = self.questions_to_improve_knowledge(x)
        knowledge_index = self.knowledge_index(x)
        information_in_indices = self.information_in_knowledge_index(x)

        attention_to_relevant_indexes = jnp.matmul(
            formulated_question, jnp.swapaxes(knowledge_index, -2, -1)
        ) / jnp.sqrt(self.brain_size)
        # mask out the future tokens
        mask = jnp.tril(jnp.ones((T, T)))  # tril = lower triangle
        attention_to_relevant_indexes = jnp.where(
            mask == 0, -jnp.inf, attention_to_relevant_indexes
        )

        probability_knowledge_index_is_relevant = nn.softmax(
            attention_to_relevant_indexes, axis=-1
        )
        answer_to_question_based_on_information_in_knowledge_index = jnp.matmul(
            probability_knowledge_index_is_relevant, information_in_indices
        )
        # return both for debugging purposes
        return (
            answer_to_question_based_on_information_in_knowledge_index,
            attention_to_relevant_indexes,
        )


class BrainStorm(nn.Module):
    n_ideas: int
    brain_size: int

    def setup(self) -> None:
        chex.assert_equal(self.brain_size % self.n_ideas, 0)
        self.idea_size = self.brain_size // self.n_ideas

        self.thoughts = [Thought(self.idea_size) for _ in range(self.n_ideas)]

        self.filter_interesting_thoughts = nn.Dense(
            features=self.brain_size,
            kernel_init=nn.initializers.xavier_uniform(),
            use_bias=False,
        )

    def __call__(self, x):
        thoughs = jnp.concatenate([thought(x)[0] for thought in self.thoughts], axis=-1)
        interesting_thoughts = self.filter_interesting_thoughts(thoughs)
        return interesting_thoughts


class ProjectIdeas(nn.Module):
    brain_size: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, training=True):
        x = nn.Dense(
            features=4 * self.brain_size,
            kernel_init=nn.initializers.xavier_uniform(),
            use_bias=False,
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            features=self.brain_size,
            kernel_init=nn.initializers.xavier_uniform(),
            use_bias=False,
        )(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not training)
        return x


class CreativityBlock(nn.Module):
    brain_size: int
    n_ideas: int
    dropout_rate: float

    def setup(self):
        # ln -> mha -> projection -> ln
        # Andrej Karpathy min 1:35:30 https://www.youtube.com/watch?v=kCc8FmEb1nY --> move layernorm before mha
        self.normalize_content1 = nn.LayerNorm()
        self.brainstrom = BrainStorm(self.n_ideas, self.brain_size)
        self.normalize_content2 = nn.LayerNorm()
        self.project_ideas = ProjectIdeas(self.brain_size, self.dropout_rate)

    def __call__(self, x, training=True):
        # lha -> brainstrom
        x = self.normalize_content1(x)
        x = self.brainstrom(x)

        # residual connection
        x = x + x

        # ln -> projection
        x = self.normalize_content2(x)
        x = self.project_ideas(x, training=training)

        # residual connection
        x = x + x
        return x


class IdeaIteration(nn.Module):
    n_moldings: int
    brain_size: int
    n_ideas: int
    dropout_rate: float

    def setup(self):
        self.creativity_blocks = [
            CreativityBlock(self.brain_size, self.n_ideas, self.dropout_rate)
            for _ in range(self.n_moldings)
        ]

    def __call__(self, x, training=True):
        for creativity_block in self.creativity_blocks:
            x = creativity_block(x, training=training)
        return x


class IdeaArticulation(nn.Module):
    max_seq_len: int
    brain_size: int
    dropout_rate: float
    chef_vocabulary_size: int

    @nn.compact
    def __call__(self, x, training=True):
        # flatten
        x = jnp.reshape(x, (x.shape[0], -1))
        x = nn.LayerNorm()(x)
        x = nn.Dense(
            features=self.brain_size, kernel_init=nn.initializers.xavier_uniform()
        )(x)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not training)
        x = nn.Dense(
            features=self.max_seq_len * self.chef_vocabulary_size,
            kernel_init=nn.initializers.xavier_uniform(),
        )(x)
        x = jnp.reshape(x, (x.shape[0], self.max_seq_len, self.chef_vocabulary_size))
        return x


class ChefBrain(nn.Module):
    max_seq_len: int
    brain_size: int
    n_ideas: int
    n_moldings: int
    dropout_rate: float
    chef_vocabulary_size: int

    @nn.compact
    def __call__(self, x, training=True):
        x = StructureInformation(
            self.max_seq_len, self.brain_size, self.chef_vocabulary_size
        )(x)
        x = IdeaIteration(
            self.n_moldings, self.brain_size, self.n_ideas, self.dropout_rate
        )(x, training=training)
        x = IdeaArticulation(
            self.max_seq_len,
            self.brain_size,
            self.dropout_rate,
            self.chef_vocabulary_size,
        )(x, training=training)
        return x


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
