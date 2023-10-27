#Author: Kwabena Aboagye-Otchere
import json

def filter_users(input_file, output_file, min_course_depth):
    """This program filters out students who don't have adequate course depth
        to train the recommender algorithm.
        It writes the output to a json file specified in the command line argument"""
    
    #open the user.json file
    with open(input_file, 'r') as infile:
        #list meant to hold the users that meet the filter criteria of k number of courses (currently 4)
        filtered_users = []

        for line in infile:
            try:
                user = json.loads(line)
                if len(user.get("course_order", [])) >= min_course_depth:
                    filtered_users.append(user)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON data in line: {line}")

    with open(output_file, 'w') as outfile:
        for user in filtered_users:
            json.dump(user, outfile, separators=(',', ':'), ensure_ascii=False)
            outfile.write('\n')

    print(f"Filtered users have been written to {output_file}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python filter_users.py <user.json> <output_file.json> <min_course_depth>")
        print("Usage: the expected input file should be the user.json file in the MOOCCube Dataset")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    min_course_depth = int(sys.argv[3])

    filter_users(input_file, output_file, min_course_depth)
