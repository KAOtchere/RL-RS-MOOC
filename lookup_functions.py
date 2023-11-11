import pandas as pd
import re

def find_teacher_rank(course_id, ):
    pass

def find_school_rank(course_id):
    pass

def find_course_rank(course_id):
    pass

def make_concept_adjacency_list(input_file):
    pattern = r'K_[^K]*'
    # Initialize a dictionary to represent relationships
    relationships = {}

    with open(input_file, 'r', encoding='utf-8') as file:
            for line in file:
                    line = line.strip()
                    if not line:
                        continue  # Skip empty lines

                    matches = re.findall(pattern, line)  # Find all "K_" fields in the line

                    if len(matches) == 2:
                        parent, son = matches  # Extract "prerequisite" and "concept" fields

                        #strip whitespace
                        parent = parent.strip()
                        son = son.strip()
                        if parent not in relationships:
                            relationships[parent] = [son]
                        else:
                            relationships[parent].append(son)

 

    relationships_dataframe = pd.DataFrame(relationships.items(), columns=['parent', 'son'])

    relationships_dataframe = relationships_dataframe.explode('son')
    


    return relationships_dataframe

# Perform depth-first search (DFS) to find paths to the root
def find_path(adjacency_list, concept):
    relationship = adjacency_list[adjacency_list['son'] == concept]
    visited = set()
    while (not relationship.empty) and (concept not in visited):
        visited.add(concept)
        concept = relationship['parent'].values[0]
        relationship = adjacency_list[adjacency_list['son'] == concept]
    return concept


def determine_parents_dimensions(adjacenct_list):
    parents = adjacenct_list['parent'][~adjacenct_list['parent'].isin(adjacenct_list['son'])] 
    parents_with_dimension = parents.reset_index(drop=True)
    parents_with_dimension = parents_with_dimension.reset_index().rename(columns={'index': 'dimension'})
    return parents_with_dimension

def find_concept_dimension(adjacency_list, parent_dimensions, concept):
    parent = find_path(adjacency_list, concept)
    parent_record = parent_dimensions[parent_dimensions['concept'] == parent]
    if parent_record.empty:
         return -1
    dimension = parent_record['dimension'].values[0]
    return dimension