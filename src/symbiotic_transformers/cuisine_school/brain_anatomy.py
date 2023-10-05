import jax
import jax.numpy as jnp
import flax.linen as nn
import chex


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
