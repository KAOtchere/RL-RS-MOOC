import re
import json
import pandas as pd

def generate_course_rankings(input_file):

    pattern = r"C_course-v1:(\w+)\+(\w+)(?:\+\w+)?"

    course_count = {}

    with open(input_file, 'r') as infile:

        for line in infile:
            try:
                user = json.loads(line)
                course_order = user.get("course_order", [])
                for course in course_order:
                    match = re.match(pattern, course)
                    if match:
                        course_id = match.groups()[1]
                        if course_id in course_count:
                            course_count[course_id] += 1
                        else:
                            course_count[course_id] = 1
            except:
                print(f"error encountered, skipping this record: {line}")

    # Step 3: Create a dataframe from the dictionary
    df = pd.DataFrame(list(course_count.items()), columns=['Course', 'Occurrences'])

    # Step 4: Sort the dataframe by the occurrences
    df_sorted = df.sort_values(by='Occurrences', ascending=False)

    df_sorted = df_sorted.reset_index()
    df_sorted = df_sorted.drop(['index'], axis=1)
    df_sorted = df_sorted.reset_index().rename(columns={'index': 'Rank'})

    return df_sorted
    

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python3 rank_courses.py <list-of-users-with-courses.json> <output-file.csv>")
        sys.exit(1)
    input_file = sys.argv[1]
    course_output_file = sys.argv[2]

    ranked_courses = generate_course_rankings(input_file)
    
    # Step 5: Write the dataframe to a CSV file
    ranked_courses.to_csv(course_output_file, index=False)
    print(f"Ranked courses written to {course_output_file}")
    
