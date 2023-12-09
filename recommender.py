import os
import numpy as np
import tensorflow as tf
from actor import Actor
from critic import Critic
from datetime import datetime
import matplotlib.pyplot as plt
from embedding import UserCourseEmbedding
from replay_buffer import PriorityExperienceReplay
from state_representation import DRRAveStateRepresentation


class RLAgent:

    def __init__(self, env, users_num, items_num, state_size, is_test=False):

        self.env = env

        self.users_num = users_num
        self.items_num = items_num

        self.embedding_dim = 100
        self.feature_dim = 224
        self.actor_hidden_dim = 128
        self.actor_learning_rate = 0.001
        self.critic_hidden_dim = 128
        self.critic_learning_rate = 0.001
        self.discount_factor = 0.9
        self.tau = 0.001

        self.replay_memory_size = 1000000
        self.batch_size = 32

        self.actor = Actor(self.embedding_dim, self.actor_hidden_dim, self.actor_learning_rate, state_size, self.tau)  # Check later
        self.critic = Critic(self.critic_hidden_dim, self.critic_learning_rate, self.embedding_dim, self.tau)  # Check later

        self.embedding_network = UserCourseEmbedding(users_num, items_num, self.embedding_dim)
        self.embedding_network([np.zeros((1,)), np.zeros((1,))])

        self.save_model_weight_dir = f"./save_model/train-{datetime.now().strftime('%Y-%m-%d-%H')}"
        if not os.path.exists(self.save_model_weight_dir):
            os.makedirs(os.path.join(self.save_model_weight_dir, 'images'))

        self.srm_ave = DRRAveStateRepresentation(self.embedding_dim)
        self.srm_ave([np.zeros((1, 100,)), np.zeros((1, state_size, 100))])

        # PriorityExperienceReplay
        self.buffer = PriorityExperienceReplay(self.replay_memory_size, self.embedding_dim)
        self.epsilon_for_priority = 1e-6

        # ε-greedy exploration hyperparameter
        self.epsilon = 1.
        self.epsilon_decay = (self.epsilon - 0.1)/500000
        self.std = 1.5

        self.is_test = is_test

    def calculate_td_target(self, rewards, q_values, dones):
        y_t = np.copy(q_values)
        for i in range(q_values.shape[0]):
            y_t[i] = rewards[i] + (1 - dones[i])*(self.discount_factor * q_values[i])
        return y_t

    def recommend_item(self, action, recommended_items, top_k=False, items_ids=None):
        if items_ids == None:
            items_ids = np.array(list(set(i for i in range(self.items_num)) - recommended_items))

        items_ebs = self.embedding_network.get_layer('course_embedding')(items_ids)

        action = tf.transpose(action, perm=(1, 0))
        if top_k:
            item_indice = np.argsort(tf.transpose(tf.keras.backend.dot(items_ebs, action), perm=(1, 0)))[0][-top_k:]
            return items_ids[item_indice]
        else:
            item_idx = np.argmax(tf.keras.backend.dot(items_ebs, action))
            return items_ids[item_idx]

    def train(self, max_episode_num, top_k=False):
        self.actor.update_target_network()
        self.critic.update_target_network()

        episodic_precision_history = []

        for episode in range(max_episode_num):
            # episodic reward and other variables reset
            episode_reward, correct_count, steps, q_loss, mean_action = 0, 0, 0, 0, 0

            # Environment reset
            user_id, items_ids, feature_vectors, done = self.env.reset()

            while not done:
                # Observe current state & Find action
                # Embedding
                # Validate items_ids just before embedding
                validated_items_ids = [item_id for item_id in items_ids if item_id < 2288]

                user_eb = self.embedding_network.get_layer('user_embedding')(np.array(user_id))
                items_eb = self.embedding_network.get_layer('course_embedding')(np.array(validated_items_ids))

                # State representation
                state = self.srm_ave([np.expand_dims(user_eb, axis=0), np.expand_dims(items_eb, axis=0)])

                # Action (ranking score)
                action = self.actor.network(state)

                # ε-greedy exploration
                if self.epsilon > np.random.uniform() and not self.is_test:
                    self.epsilon -= self.epsilon_decay
                    action += np.random.normal(0, self.std, size=action.shape)

                # Item recommendation
                recommended_item = self.recommend_item(action, self.env.recommended_items, top_k=top_k)

                # Calculate reward & observe new state (in env)
                next_items_ids, reward, done, recommended_items, next_feature_vectors = self.env.step(recommended_item, top_k=top_k)

                if top_k:
                    reward = np.sum(reward)

                # Validate next_items_ids just before embedding
                validated_next_items_ids = [item_id for item_id in next_items_ids if item_id < 2288]

                # Get next_state
                next_items_eb = self.embedding_network.get_layer('course_embedding')(np.array(validated_next_items_ids))
                next_state = self.srm_ave([np.expand_dims(user_eb, axis=0), np.expand_dims(next_items_eb, axis=0)])

                # Buffer update
                self.buffer.append(state, action, reward, next_state, done)

                # Buffer replay logic
                # ... existing buffer replay and network update logic ...

                items_ids = validated_next_items_ids  # Update items_ids for next iteration
                episode_reward += reward
                mean_action += np.sum(action[0]) / len(action[0])
                steps += 1

                if reward > 0:
                    correct_count += 1

                print(
                    f'recommended items : {len(self.env.recommended_items)},  epsilon : {self.epsilon:0.3f}, reward : {reward:+}',
                    end='\r')

                if done:
                    # Print and record episode summary
                    print()
                    precision = int(correct_count / steps * 100)
                    print(
                        f'{episode}/{max_episode_num}, precision : {precision:2}%, total_reward:{episode_reward}, q_loss : {q_loss / steps}, mean_action : {mean_action / steps}')
                    episodic_precision_history.append(precision)

            if (episode+1) % 50 == 0:
                plt.plot(episodic_precision_history)
                plt.savefig(os.path.join(self.save_model_weight_dir, f'images/training_precision_%_top_5.png'))

            if (episode+1) % 1000 == 0 or episode == max_episode_num-1:
                self.save_model(os.path.join(self.save_model_weight_dir, f'actor_{episode+1}_fixed.h5'),
                                os.path.join(self.save_model_weight_dir, f'critic_{episode+1}_fixed.h5'))

    def save_model(self, actor_path, critic_path):
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)

    def load_model(self, actor_path, critic_path):
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)