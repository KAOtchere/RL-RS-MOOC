#Author: Kwabena Aboagye-Otchere
import json
import csv
import sys

def extract_distinct_courses(input_file, output_file):
    """This program extracts the distinct courses 
        that exist in the user profiles after filtering them. 
        It writes to a CSV file specified in the command line arguments.
        You may use the JSON file generated from running the user-cleaner program
        or the user.json file from the MOOCCube dataset."""
    
    courses = set()

    #open the filtered user json file generated from user-cleaner program
    with open(input_file, 'r') as infile:
        for line in infile:
            try:
                #read user object and extract courses from course order field
                user = json.loads(line)
                
                #add new courses found to set
                courses.update(user.get('course_order', []))
            except json.JSONDecodeError:
                print(f"Failed to parse JSON data in line: {line}")

    #write extracted courses to csv file
    with open(output_file, 'w', newline='') as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(['Course'])  # Write a header row

        for course in courses:
            csv_writer.writerow([course])

    print(f"Distinct courses have been written to {output_file}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python extract_courses.py <filtered-users-input-file.json> <output_file.csv>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    extract_distinct_courses(input_file, output_file)
