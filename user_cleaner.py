#Author: Kwabena Aboagye-Otchere
import pandas as pd

def filter_users(input_file, min_course_depth):
    users = pd.read_json(input_file, lines=True, encoding='utf-8')
    
    filtered_users = users[users['course_order'].apply(len) >= min_course_depth]

    return filtered_users
    
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python filter_users.py <user.json> <output_file.json> <min_course_depth>")
        print("Usage: the expected input file should be the user.json file in the MOOCCube Dataset")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    min_course_depth = int(sys.argv[3])

    filtered_users = filter_users(input_file, min_course_depth)

    filtered_users.to_json(output_file, lines=True, orient='records', force_ascii=False)

