import re
import json
import pandas as pd

def generate_school_rankings(input_file):

    pattern = r"C_course-v1:(\w+)\+(\w+)(?:\+\w+)?"

    school_count = {}

    with open(input_file, 'r') as infile:

        for line in infile:
            try:
                user = json.loads(line)
                course_order = user.get("course_order", [])
                for course in course_order:
                    match = re.match(pattern, course)
                    if match:
                        institution = match.groups()[0]

                        if institution in school_count:
                            school_count[institution] += 1
                        else:
                            school_count[institution] = 1
            except:
                print(f"error encountered, skipping this record: {line}")

    

    # Step 3: Create a dataframe from the dictionary
    df = pd.DataFrame(list(school_count.items()), columns=['School', 'Occurrences'])

    # Step 4: Sort the dataframe by the occurrences
    df_sorted = df.sort_values(by='Occurrences', ascending=False)

    df_sorted = df_sorted.reset_index()
    df_sorted = df_sorted.drop(['index'], axis=1)
    df_sorted = df_sorted.reset_index().rename(columns={'index': 'Rank'})

    return df_sorted


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python3 rank_schools.py <list-of-users-with-courses.json> <output-file.csv>")
        sys.exit(1)
    input_file = sys.argv[1]
    school_output_file = sys.argv[2]

    ranked_schools = generate_school_rankings(input_file)
    ranked_schools.to_csv(school_output_file, index=False)
    print(f"Ranked schools written to {school_output_file}")

    
