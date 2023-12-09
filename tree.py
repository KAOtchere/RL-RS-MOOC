import numpy as np

# Classes used for the Prioritized Experience Replay algorithm


class SumTree:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size  # Number of leaf nodes (where data is stored)
        self.tree = np.zeros((buffer_size * 2 - 1))  # Binary tree as a numpy array
        self.index = buffer_size - 1  # Start index for the leaf nodes

    def update_tree(self, index):
        # Update the tree whenever a new priority is added or an existing one is changed
        while True:
            index = (index - 1) // 2  # Move to the parent node
            left = (index * 2) + 1  # Index of left child
            right = (index * 2) + 2  # Index of right child
            self.tree[index] = self.tree[left] + self.tree[right]  # Parent node's value is the sum of its children
            if index == 0:
                break  # Stop if we reach the root

    def add_data(self, priority):
        # Add a new priority to the tree
        if self.index == self.buffer_size * 2 - 1:
            self.index = self.buffer_size - 1  # Reset index if it reaches the end of the array

        self.tree[self.index] = priority  # Set the priority in the leaf node
        self.update_tree(self.index)  # Update the tree with this new priority
        self.index += 1  # Move to the next index

    def search(self, num):
        # Search for a leaf node with a value <= num
        current = 0  # Start from the root
        while True:
            left = (current * 2) + 1  # Index of left child
            right = (current * 2) + 2  # Index of right child

            if num <= self.tree[left]:
                current = left  # Go to left child if num is less than or equal to its value
            else:
                num -= self.tree[left]  # Subtract the left child's value from num and go to right child
                current = right

            if current >= self.buffer_size - 1:
                break  # Stop if we reach a leaf node

        return self.tree[current], current, current - self.buffer_size + 1  # Return the leaf node's value, its index, and the index relative to the start of the leaf nodes

    def update_priority(self, priority, index):
        # Update a priority at a specific index
        self.tree[index] = priority  # Set the new priority
        self.update_tree(index)  # Update the tree based on this new priority

    def sum_all_priority(self):
        # Return the sum of all priorities
        return float(self.tree[0])  # The sum of all priorities is stored at the root of the tree


class MinTree:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size  # Number of leaf nodes
        self.tree = np.ones((buffer_size * 2 - 1))  # Binary tree as a numpy array, initialized with ones
        self.index = buffer_size - 1  # Start index for the leaf nodes

    def update_tree(self, index):
        # Update the tree whenever a new priority is added or an existing one is changed
        while True:
            index = (index - 1) // 2  # Move to the parent node
            left = (index * 2) + 1  # Index of left child
            right = (index * 2) + 2  # Index of right child
            # Parent node's value is the minimum of its children
            self.tree[index] = np.minimum(self.tree[left], self.tree[right])
            if index == 0:
                break  # Stop if we reach the root

    def add_data(self, priority):
        # Add a new priority to the tree
        if self.index == self.buffer_size * 2 - 1:
            self.index = self.buffer_size - 1  # Reset index if it reaches the end

        self.tree[self.index] = priority  # Set the priority in the leaf node
        self.update_tree(self.index)  # Update the tree with this new priority
        self.index += 1  # Move to the next index

    def update_priority(self, priority, index):
        # Update a priority at a specific index
        self.tree[index] = priority  # Set the new priority
        self.update_tree(index)  # Update the tree based on this new priority

    def min_priority(self):
        # Return the minimum priority
        return float(self.tree[0])  # The minimum priority is stored at the root of the tree

