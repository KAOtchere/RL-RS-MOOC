import tensorflow as tf


class DRRAveStateRepresentation(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(DRRAveStateRepresentation, self).__init__()
        self.embedding_dim = embedding_dim
        self.wav = tf.keras.layers.Conv1D(1, 1, 1)
        self.concat = tf.keras.layers.Concatenate()
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs, **kwargs):
        items_eb = tf.transpose(inputs[1], perm=(0, 2, 1)) / self.embedding_dim

        # Adjust the padding logic if necessary
        if items_eb.shape[-1] != 4:
            if items_eb.shape[-1] > 4:
                items_eb = items_eb[..., :4]
            else:
                padding = tf.zeros((items_eb.shape[0], items_eb.shape[1], 4 - items_eb.shape[-1]))
                items_eb = tf.concat([items_eb, padding], axis=-1)

        wav = self.wav(items_eb)
        wav = tf.transpose(wav, perm=(0, 2, 1))
        wav = tf.squeeze(wav, axis=1)
        user_wav = tf.keras.layers.multiply([inputs[0], wav])
        concat = self.concat([inputs[0], user_wav, wav])
        return self.flatten(concat)