import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from lookup_functions import find_course_vector
from lookup_functions import find_course_rank

learning_rate = 0.00002
gamma = 0.1
lam_bda = 0.95
epsilon = 0.1
n_epochs = 4
n_rollout = 500

course_dictionary = pd.read_csv('final_courses.csv', header=0)

class PPO(nn.Module):
    def __init__(self, input_size, output_size, courses_feature_vector) -> None:
        super(PPO, self).__init__()
        self.data = []

        self.courses_feature_vector = courses_feature_vector

        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.fc7 = nn.Linear(32, 16)

        self.fc_pi = nn.Linear(16, output_size)
        self.fc_v = nn.Linear(16, 1)

        self.optimiser = optim.Adam(self.parameters(), lr=learning_rate)

    
    def pi(self, user, softmax_dim = 0):
        course_vectors = list()

        for course_id in user.courses:
            course = find_course_vector(course_id, course_dictionary)
            if course != None:
                concept_vector = np.array(course['concepts_vector'])
                course_rank = course['course_rank']
                school_rank = course['school_rank']
                teacher_rank = course['teacher_rank']
                ranks = np.array([course_rank, school_rank, teacher_rank])
                course_vector = np.concatenate(ranks, concept_vector)
                course_vectors.append(course_vector)

        course_vectors = np.asarray(course_vectors)

        average_of_vectors = np.mean(course_vectors, axis=0)

        x = average_of_vectors

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        x = self.fc_pi(x)

        prob = F.log_softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, user):
        course_vectors = list()

        for course_id in user.courses:
            course = find_course_vector(course_id, course_dictionary)
            if course != None:
                concept_vector = np.array(course['concepts_vector'])
                course_rank = course['course_rank']
                school_rank = course['school_rank']
                teacher_rank = course['teacher_rank']
                ranks = np.array([course_rank, school_rank, teacher_rank])
                course_vector = np.concatenate(ranks, concept_vector)
                course_vectors.append(course_vector)

        course_vectors = np.asarray(course_vectors)

        average_of_vectors = np.mean(course_vectors, axis=0)

        x = average_of_vectors

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        v = self.fc_v(x)

        return v
    
    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        
        

        

    