import pandas as pd
import json
import sys
from lookup_functions import make_concept_adjacency_list, find_path, determine_parents_dimensions, find_concept_dimension, count_sort
#for each course in courses.csv, we make the vector representation for it

ranked_courses = pd.read_csv('ranked_courses.csv', header=0)
ranked_schools = pd.read_csv('ranked_Schools.csv', header=0)
ranked_teachers = pd.read_csv('ranked_teachers.csv', header=0)

course_teachers = pd.read_csv('teacher-course.json',  sep='\t', header=None, names=['Teacher', 'Course'])

users = pd.read_json('filtered-users.json', lines=True, encoding='utf-8')

all_courses = pd.read_json('courses-with-c-and-p.json', lines=True, encoding='utf-8')
all_courses['school'] = all_courses['course'].str.extract(r'C_course-v1:(\w+)')
all_courses['course_id'] = all_courses['course'].str.extract(r'C_course-v1:\w+\+(\w+)')

all_courses = pd.merge(all_courses, ranked_courses, how='left', left_on='course_id', right_on='Course')
all_courses = all_courses.drop(axis=1, columns=['Course', 'Occurrences'])
all_courses.rename(columns={'rank': 'course_rank'}, inplace=True)

all_courses = pd.merge(all_courses, ranked_schools, how='left', left_on='school', right_on='School')
all_courses = all_courses.drop(axis=1, columns=['School', 'Occurrences'])
all_courses.rename(columns={'rank': 'school_rank'}, inplace=True)

all_courses = pd.merge(all_courses, course_teachers, how='left', left_on='course', right_on='Course')
all_courses = all_courses.drop(axis=1, columns=['Course'])

all_courses = pd.merge(all_courses, ranked_teachers, how='left', on='Teacher')
all_courses = all_courses.drop(axis=1, columns=['Occurrences'])

all_courses.rename(columns={'Rank': 'teacher_rank'}, inplace=True)

concept_graph = make_concept_adjacency_list('prerequisite-dependency.json')
parent_concepts = determine_parents_dimensions(concept_graph)

vector_size = len(parent_concepts)

# all_courses = all_courses.head(20)

all_courses['encoded_concepts'] = all_courses['concepts'].apply(lambda lst: [find_concept_dimension(v, concept_graph, parent_concepts) for v in lst])
all_courses['concepts_vector'] = all_courses['encoded_concepts'].apply(count_sort, dimension_size=vector_size)

all_courses['encoded_prerequisites'] = all_courses['prerequsities'].apply(lambda lst: [find_concept_dimension(v, concept_graph, parent_concepts) for v in lst])
all_courses['prerequisites_vector'] = all_courses['encoded_prerequisites'].apply(count_sort, dimension_size=vector_size)




#write dataframe to file
all_courses.to_csv('final_courses.csv')