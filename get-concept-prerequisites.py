#Author: Kwabena Aboagye-Otchere

import re
import json
import sys

# This pattern allows us to get the concepts in the prerequisite-dependent relation file.
# The format for each relation is <K_prerequisite> <K_concept> but each of these arguments may have whitespace within them
pattern = r'K_[^K]*'

def extract_relationships(courses_file, prerequisite_dependency_file, output_file):
    # Create a dictionary to store a concept and its prerequisites
    relationship_dict = {}
    with open(prerequisite_dependency_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            matches = re.findall(pattern, line)  # Find all "K_" fields in the line

            if len(matches) == 2:
                prereq, dep = matches  # Extract "prerequisite" and "concept" fields

                #strip whitespace
                prereq = prereq.strip()
                dep = dep.strip()

                if dep not in relationship_dict:
                    relationship_dict[dep] = set()  # Create an empty set for each "concept" field that has not been encountered yet

                relationship_dict[dep].add(prereq)  # Add the newly found "prerequisite" to the set for the corresponding concept
                
    with open(courses_file, 'r', encoding='utf-8') as courses_file:
        #stores the course objects after appending the prerequisite object field
        updated_courses = []

        for line in courses_file: #read the file with objects that has courses and concept fields
            try:
                course = json.loads(line)
                prerequisites = set() #instantiate prerequisite set for the courses
                concepts = course.get("concepts", []) #read the concepts as type array

                #lookup each concept in the relationship dictionary extracted earlier and update course prerequisites with prerequisite for each concept
                for concept in concepts:
                    if concept in relationship_dict:
                        prerequisites.update(list(relationship_dict[concept]))
                course["prerequsities"] = list(prerequisites) #add prerequisite field to course object
                updated_courses.append(course) #add course to new course data structure

            except json.JSONDecodeError:
                print(f"Failed to parse JSON data in line: {line}")


    # Write the JSON data to the output file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for course in updated_courses:
            json.dump(course, outfile, separators=(',', ':'), ensure_ascii=False)
            outfile.write('\n')

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python get-concept-prerequisites.py <courses.json> <prerequisite-dependency.json> <output.json>")
        sys.exit(1)
    courses_file = sys.argv[1]
    prerequisite_dependency_file = sys.argv[2]
    output_file = sys.argv[3]

    extract_relationships(courses_file, prerequisite_dependency_file, output_file)
    print(f"Relationships extracted and saved to '{output_file}'")
