import ast
import numpy as np
from sklearn.preprocessing import StandardScaler


class OfflineEnv(object):

    def __init__(self, users, courses, state_size, fix_user_id=None):
        # Initialization of the environment
        self.users = users  # Dataset of users
        self.courses = courses  # Dataset of courses
        self.state_size = state_size  # Maximum number of items in the current recommendation state
        self.available_users = self.users  # Set of users available for recommendation
        self.fix_user_id = fix_user_id  # Optionally fix the user ID for the recommendation process

        # Initialize attributes
        self.user = None
        self.selected_user_idx = None
        self.course_ranks = None
        self.school_ranks = None
        self.teacher_ranks = None
        self.feature_vectors = None
        self.items = []
        self.user_items = []
        self.concept_vectors = []

        self.done = False
        self.recommended_items = set()
        self.done_count = 3000

        # Initialize user state
        self.initialize_user_state()

    def initialize_user_state(self):
        # Selecting a user for the recommendation process
        self.user = self.fix_user_id if self.fix_user_id else self.available_users.sample(n=1).iloc[0]
        self.selected_user_idx = self.fix_user_id if self.fix_user_id else self.user['idx']

        # Preparing the list of courses and concept vectors based on the selected user's preferences
        self.user_items = []
        self.concept_vectors = []
        self.course_ranks = []
        self.school_ranks = []
        self.teacher_ranks = []

        for course_id in self.user['course_order']:
            course_match = self.courses[self.courses['course'] == course_id]
            if not course_match.empty:
                self.user_items.append(course_match['idx'].values[0])

                # Fetch the concept vector
                if 'concepts_vector' in course_match.columns:
                    self.concept_vectors.append(course_match['concepts_vector'].values[0])
                else:
                    self.concept_vectors.append(None)  # or some default value

                # Fetch course rank, school rank, and teacher rank
                self.course_ranks.append(course_match['course_rank'].values[0])
                self.school_ranks.append(course_match['school_rank'].values[0])
                self.teacher_ranks.append(course_match['teacher_rank'].values[0])

        # Initialize the items for the recommendation state
        self.items = np.array(self.user_items[:self.state_size])  # course ids
        self.concept_vectors = self.concept_vectors[: self.state_size]  # concepts
        self.course_ranks = np.array(self.course_ranks[: self.state_size])  # course ranks by popularity
        self.school_ranks = np.array(self.school_ranks[: self.state_size])  # school ranks by popularity
        self.teacher_ranks = np.array(self.teacher_ranks[: self.state_size])  # teachers ranks by popularity

        # Steps to process the concept vectors
        # Convert string representations to actual lists using ast.literal_eval
        self.concept_vectors = [ast.literal_eval(vec) if isinstance(vec, str) else vec for vec in self.concept_vectors]
        self.concept_vectors = np.array([np.array(vec) for vec in self.concept_vectors], dtype=np.float32)

        # Handle NaN values by replacing them with the mean of the non-NaN elements
        if np.isnan(self.teacher_ranks).any():
            non_nan_mean = np.nanmean(self.teacher_ranks)
            self.teacher_ranks = np.where(np.isnan(self.teacher_ranks), non_nan_mean, self.teacher_ranks)

        # Average vectors before concatenating them
        self.concept_vectors = np.mean(self.concept_vectors, axis=0)
        self.course_ranks = np.mean(self.course_ranks, axis=0)
        self.school_ranks = np.mean(self.school_ranks, axis=0)
        self.teacher_ranks = np.mean(self.teacher_ranks, axis=0)

        # Normalize the ranks
        # First, put the ranks into an array
        ranks = np.array([self.course_ranks, self.school_ranks, self.teacher_ranks])
        scaler = StandardScaler()
        # Reshape ranks to 2D array as StandardScaler expects 2D inputs
        normalized_ranks = scaler.fit_transform(ranks.reshape(-1, 1)).flatten()

        # Concatenate the concept vectors with normalized ranks
        self.feature_vectors = np.concatenate((self.concept_vectors.reshape(1, -1), normalized_ranks.reshape(1, -1)), axis=1).flatten()

        self.done = False  # Flag to check if the recommendation process is complete
        self.recommended_items = set(self.items)  # Set of items already recommended
        self.done_count = 3000  # Threshold for completing the recommendation process

    def reset(self):
        # Reset the environment to start a new recommendation episode
        self.initialize_user_state()
        return self.selected_user_idx, self.items, self.feature_vectors, self.done

    def step(self, action, top_k=False):
        # Process a step in the environment given an action (course recommendation)
        reward = -0.5  # Default penalty for incorrect recommendation

        if top_k:
            # If handling multiple recommendations at once
            correctly_recommended = []
            rewards = []
            for act in action:
                # Check if each action is a correct recommendation
                if act in self.user_items and act not in self.recommended_items:
                    correctly_recommended.append(act)
                    rewards.append(1)  # Reward for correct recommendation
                else:
                    rewards.append(-1)  # Penalty for incorrect recommendation
                self.recommended_items.add(act)
            if max(rewards) > 0:
                self.items = self.items + correctly_recommended
            reward = rewards
        else:
            # For a single recommendation
            if action in self.user_items and action not in self.recommended_items:
                reward = 1  # Reward for correct single recommendation
            if reward > 0:
                self.items = self.items + [action]
            self.recommended_items.add(action)

        # Check if the recommendation process is complete
        if len(self.recommended_items) > self.done_count or (len(self.recommended_items) >= len(self.user_items)):
            self.done = True

        return self.items, reward, self.done, self.recommended_items, self.feature_vectors

    def get_items_names(self):
        # Retrieve the names of the courses from their indices
        items_names = self.courses['idx'].map(dict(zip(self.courses['idx'], self.courses['course'])))
        return items_names
