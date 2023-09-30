import tensorflow as tf
from tasks import chef_config


def clean_and_to_char(x):
    characters = tf.strings.unicode_split(x, input_encoding="UTF-8")
    # keep only letters
    characters = tf.strings.regex_replace(characters, "[^a-zA-Z ]", "")
    # to lower case
    characters = tf.strings.lower(characters)
    return characters


def to_batch(x, window_size, batch_size):
    max_value = tf.cast(tf.shape(x)[0] - window_size, tf.int64)
    idxs = tf.experimental.numpy.random.randint(0, max_value, size=batch_size)

    # some weird tf stuff...
    def slice_sequence(idx):
        return x[idx : idx + window_size]

    batch = tf.map_fn(slice_sequence, idxs, dtype=tf.int64)
    return batch


def imitate_chefs(text: str, config: chef_config.ChefConfig) -> [tf.data.Dataset]:
    # clean dataset
    imitation_ds = tf.data.Dataset.from_tensor_slices([text])
    imitation_ds = imitation_ds.map(clean_and_to_char)

    # fit stoi and itos
    stoi = tf.keras.layers.StringLookup(mask_token="|", oov_token="?")
    stoi.adapt(imitation_ds)
    itos = tf.keras.layers.StringLookup(
        vocabulary=stoi.get_vocabulary(), invert=True, mask_token="|", oov_token="?"
    )

    # turn into batches
    imitation_ds = imitation_ds.map(stoi)
    imitation_ds = imitation_ds.map(
        lambda x: to_batch(
            x, config.max_seq_len + 1, config.num_subordinates_per_shop * config.n_shops
        )
    )
    imitation_ds = imitation_ds.map(lambda x: (x[:, :-1], x[:, 1:]))
    return imitation_ds.repeat().prefetch(tf.data.AUTOTUNE), stoi, itos


def decode(x, itos):
    return tf.strings.reduce_join(itos(x), axis=1)


if __name__ == "__main__":
    # example of how to use the recipe reader
    path_to_text_file = "./recipes/recipes.txt"
    number_of_characters = 10000
    text = (
        open(path_to_text_file, "rb")
        .read()
        .decode(encoding="utf-8")[:number_of_characters]
    )
    imitation_ds, stoi, itos = imitate_chefs(text, chef_config.ChefConfig())
    for x, y in imitation_ds.take(2):
        print(x.shape)
        print(y.shape)
        print(decode(x, itos)[0])
