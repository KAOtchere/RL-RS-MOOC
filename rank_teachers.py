import json
import pandas as pd
import sys


def generate_teacher_rankings(users_data, teacher_course_data):
    user_courses = list()
    # Step 1: Load the users
    with open(users_data, 'r', encoding='utf-8') as file:
        # users_data = [json.loads(line) for line in file]

    # Extract a list of all courses taken by users
    # user_courses = [course for user in users_data for course in user['course_order']]
        
        for record in file:
            user = json.loads(record)
            user_courses.extend(user.get('course_order', []))
    
    # Step 2: Load the teacher-course pairs from File 2
    # Create a mapping of courses to teachers
    course_teacher_dict = {}
    all_teachers = set()
    with open(teacher_course_data, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            teacher, course = line.split('\t')
            all_teachers.add(teacher)
            if course in course_teacher_dict:
                course_teacher_dict[course].append(teacher)
            else:
                course_teacher_dict[course] = [teacher]

    # Step 3: Count how many times each teacher appears
    teacher_count = {}
    counted_teachers = set()
    for course in user_courses:
        if course in course_teacher_dict:
            for teacher in course_teacher_dict[course]:
                teacher_count[teacher] = teacher_count.get(teacher, 0) + 1
                counted_teachers.add(teacher)

    # If there are teachers who didn't appear in the users course history, give them a default count of 0
    uncounted_teachers = all_teachers - counted_teachers
    for teacher in uncounted_teachers:
        teacher_count.setdefault(teacher, 0)

    # Convert the count dictionary to a list of (teacher, count) pairs
    teacher_count_list = list(teacher_count.items())

    # Step 4: Rank by occurrences
    teacher_count_list.sort(key=lambda x: x[1], reverse=True)

    # Step 5: Convert to DataFrame for nicer display and possible further manipulation
    df_teachers = pd.DataFrame(teacher_count_list, columns=['Teacher', 'Occurrences'])

    # Add a 'Rank' column
    df_teachers['Rank'] = df_teachers['Occurrences'].rank(method='max', ascending=False).astype(int)

    # Make 'Rank' the first column
    df_teachers = df_teachers[['Rank', 'Teacher', 'Occurrences']]

    # Sort the DataFrame based on 'Rank'
    df_teachers.sort_values(by='Rank', inplace=True)

    # Reset index if you want to remove the old index
    df_teachers.reset_index(drop=True, inplace=True)
    
    return df_teachers
    


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python teachers_rank.py <user.json> <teacher-course.txt> <output_file>")
        print("Usage: the expected input file should be the user.json file and the teacher-course.json in the MOOCCube Dataset")
        sys.exit(1)

    users_data = sys.argv[1]
    teacher_course_file = sys.argv[2]
    output_file = sys.argv[3]
    ranked_teachers = generate_teacher_rankings(users_data, teacher_course_file)
    ranked_teachers.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Ranked teachers written to {output_file}")
    