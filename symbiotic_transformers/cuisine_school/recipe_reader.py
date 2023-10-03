import os
import tensorflow as tf
from symbiotic_transformers.cuisine_school import chef_gpt_brain


class ImitateChefDataset:
    def __init__(self, brain_assistant: chef_gpt_brain.BrainAssistant):
        self.brain_assistant = brain_assistant

    def to_batch(self, x, window_size, batch_size):
        max_value = tf.cast(tf.shape(x)[0] - window_size, tf.int64)
        idxs = tf.experimental.numpy.random.randint(0, max_value, size=batch_size)

        # some weird tf stuff...
        def slice_sequence(idx):
            return x[idx : idx + window_size]

        batch = tf.map_fn(slice_sequence, idxs, dtype=tf.int64)
        return batch

    def _build_recipe_ds(self, text: str, is_train):
        imitation_ds = tf.data.Dataset.from_tensor_slices([text])

        if is_train:
            # fit
            fit_ds = imitation_ds.map(
                self.brain_assistant._clean_text_and_convert_to_characters
            )
            self.brain_assistant._build_chef_vocabulary(fit_ds)
        # transform
        imitation_ds = imitation_ds.map(
            self.brain_assistant.transform_recipe_for_training
        )
        imitation_ds = imitation_ds.map(
            lambda x: self.to_batch(
                x,
                self.brain_assistant.config.max_seq_len + 1,
                self.brain_assistant.config.num_subordinates_per_shop
                * self.brain_assistant.config.n_shops,
            )
        )
        imitation_ds = imitation_ds.map(lambda x: (x[:, :-1], x[:, 1:]))

        # convert to one hot labels
        imitation_ds = imitation_ds.map(
            lambda x, y: (
                x,
                tf.one_hot(y, int(self.brain_assistant.stoi.vocabulary_size())),
            )
        )

        return (
            imitation_ds.take(
                self.brain_assistant.config.n_training_recipes_per_imitation
                if is_train
                else self.brain_assistant.config.n_exam_recipes_per_imitation
            )
            .repeat()
            .prefetch(tf.data.AUTOTUNE)
            .as_numpy_iterator()
        )

    def get_recipe_datasets(self, path_to_text_file: str):
        # read in text
        with open(path_to_text_file, "r") as f:
            text = " ".join(f.readlines())[
                : self.brain_assistant.config.overfit_size_by_reducing_text
            ]
        # split into train and eval
        train_text, eval_text = (
            text[: int(len(text) * 0.9)],
            text[int(len(text) * 0.9) :],
        )

        train_ds = self._build_recipe_ds(train_text, is_train=True)
        eval_ds = self._build_recipe_ds(eval_text, is_train=False)

        if not os.path.exists(self.brain_assistant.config.vocab_dir):
            os.makedirs(self.brain_assistant.config.vocab_dir, exist_ok=True)
        self.brain_assistant.stoi.save_assets(self.brain_assistant.config.vocab_dir)
        return train_ds, eval_ds
