# Dependencies
import time
import pandas as pd
from envs import OfflineEnv  # Custom module for the offline environment
from recommender import RLAgent  # Custom module for the RL Agent

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Define root directory and data directory
ROOT_DIR = os.getcwd()  # Get the current working directory
DATA_DIR = os.path.join(ROOT_DIR, 'ml-1m')  # Define the path to the data directory

# Constants for environment and training setup
STATE_SIZE = 5  # Size of the state representation
MAX_EPISODE_NUM = 20000  # Maximum number of episodes for training

# Main execution block
if __name__ == "__main__":
    print('Data loading...')

    # Loading datasets
    # Load user data from a JSON file
    users = pd.read_json(os.path.join(DATA_DIR, 'filtered-users.json'), lines=True, encoding='utf-8')

    # Load course data from a CSV file
    courses = pd.read_csv(os.path.join(DATA_DIR, 'final_courses.csv'), header=0)

    print("Data loading complete!")
    print("Data preprocessing...")

    # Training data preparation
    train_users_num = int(len(users) * 0.8)  # Calculate 80% of the total users for training. 100% = 55203, 80% = 44162
    train_items_num = int(len(courses))  # Total number of courses = 2288

    # Sample 80% of users randomly for training
    train_user_data = users.sample(frac=0.8)

    # Reset index to renumber rows from 0, and drop the old index
    train_user_data = train_user_data.reset_index(drop=True)

    # Convert the index to a regular column
    train_user_data['idx'] = train_user_data.index

    # Now 'idx' column will have the new index values starting from 0

    print('DONE!')
    time.sleep(2)  # Wait for 2 seconds

    # Initialize the environment and the recommender system agent
    env = OfflineEnv(train_user_data, courses, STATE_SIZE)  # Create an offline environment with the training data
    recommender = RLAgent(env, len(train_user_data), len(courses), STATE_SIZE)  # Initialize the RL agent with the environment

    # Build networks for the recommender system
    recommender.actor.build_networks()  # Build the actor network
    recommender.critic.build_networks()  # Build the critic network

    # Start training the recommender system
    recommender.train(MAX_EPISODE_NUM, top_k=True)  # Train with the specified number of episodes

