import tensorflow as tf


class UserCourseEmbedding(tf.keras.Model):
    def __init__(self, len_users, len_courses, embedding_dim):
        super(UserCourseEmbedding, self).__init__()
        self.m_u_input = tf.keras.layers.InputLayer(name='input_layer', input_shape=(2,))
        # embedding
        self.u_embedding = tf.keras.layers.Embedding(name='user_embedding', input_dim=len_users,
                                                     output_dim=embedding_dim)
        self.c_embedding = tf.keras.layers.Embedding(name='course_embedding', input_dim=len_courses,
                                                     output_dim=embedding_dim)
        # dot product
        self.c_u_merge = tf.keras.layers.Dot(name='course_user_dot', normalize=False, axes=1)
        # output
        self.c_u_fc = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, **kwargs):
        x = self.m_u_input(inputs)
        user_emb = self.u_embedding(x[0])
        course_emb = self.c_embedding(x[1])
        m_u = self.c_u_merge([course_emb, user_emb])
        return self.c_u_fc(m_u)


    # class UserCourseEmbedding(tf.keras.Model):
    #     def __init__(self, len_users, len_courses, embedding_dim, feature_dim=224):
    #         super(UserCourseEmbedding, self).__init__()
    #         self.m_u_input = tf.keras.layers.InputLayer(name='input_layer', input_shape=(2 + feature_dim,))
    #         self.u_embedding = tf.keras.layers.Embedding(name='user_embedding', input_dim=len_users,
    #                                                      output_dim=embedding_dim)
    #         self.c_embedding = tf.keras.layers.Embedding(name='course_embedding', input_dim=len_courses,
    #                                                      output_dim=embedding_dim)
    #         self.feature_process = tf.keras.Sequential([
    #             tf.keras.layers.Dense(embedding_dim, activation='relu'),
    #             tf.keras.layers.Dense(embedding_dim, activation='relu')
    #         ])
    #         self.fc = tf.keras.Sequential([
    #             tf.keras.layers.Dense(embedding_dim, activation='relu'),
    #             tf.keras.layers.Dense(1, activation='sigmoid')
    #         ])
    #
    #     def call(self, inputs):
    #         user_id, course_id, features = inputs[:, 0], inputs[:, 1], inputs[:, 2:]
    #         user_emb = self.u_embedding(user_id)
    #         course_emb = self.c_embedding(course_id)
    #         feature_emb = self.feature_process(features)
    #
    #         combined = tf.concat([user_emb, course_emb, feature_emb], axis=1)
    #         return self.fc(combined)