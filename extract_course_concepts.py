#Author: Kwabena Aboagye-Otchere

import csv
import json
import sys

def create_course_concept_json(csv_file, json_file):
    """This program creates a course object that contains 
        a course id and the concepts taught in that course.
        It is generated using the course CSV file generated
        from the distinct-course-extractor program and
        the course-concept JSON file in the MOOCCube Dataset"""
    
    # Read the CSV file to get a set of courses
    courses = set()
    with open(csv_file, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)

        # Skip the header row
        next(csv_reader, None)
        
        for row in csv_reader:
            courses.add(row[0])  # Assuming the course name is in the first column

    # Read the JSON file with concepts
    course_concepts = {}
    with open(json_file, 'r', encoding='utf-8') as jsonfile:
        for line in jsonfile:
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            try:
                data = line.split()
                if isinstance(data, list) and len(data) == 2:
                    course, concept = data[0], data[1]
                    if course in courses:
                        if course not in course_concepts:
                            course_concepts[course] = []
                        course_concepts[course].append(concept)
                elif isinstance(data, list) and len(data) > 2: #if the concept is a multi-word
                    #concatenate concept into a single string with join
                    course, concept = data[0], " ".join(data[1:])
                    if course in courses:
                        if course not in course_concepts:
                            course_concepts[course] = []
                        course_concepts[course].append(concept)
                else:
                    print(f"Ignored invalid JSON data: {line}")
            except json.JSONDecodeError:
                print(f"Failed to parse JSON data in line: {line}")

    # Create the JSON object in the desired format
    output_data = [{'course': course, 'concepts': concepts} for course, concepts in course_concepts.items()]

    return output_data
    

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python create_course_concept_json.py <courses_input_file.csv> <course-concept.json> <output_file.json>")
        sys.exit(1)

    csv_file = sys.argv[1]
    json_file = sys.argv[2]
    output_file = sys.argv[3]

    courses_with_concepts = create_course_concept_json(csv_file, json_file)

    # Write the JSON data to the output file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for course_concept in courses_with_concepts:
            json.dump(course_concept, outfile, separators=(',', ':'), ensure_ascii=False)
            outfile.write('\n')

    print(f"Courses with their concepts have been written to {output_file}")
