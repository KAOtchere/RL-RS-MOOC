#Author: Kwabena Aboagye-Otchere
import json
import csv
import sys

def generate_value_report(input_file, output_file):
    """
    This program counts the number of times a concept is either a prerequisite or a concept
    within the dataset of courses passed as a command line argument.
    The courses dataset should have been generated with the get-concept-prerequisites program.
    """
    # Create a dictionary to store the value occurrences
    value_counts = {}

    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            obj = json.loads(line)
            concepts = obj.get("concepts", [])
            prerequisites = obj.get("prerequisites", [])

            # Process concepts and prerequisites to count value occurrences
            values = concepts + prerequisites
            unique_values = set(values)

            for value in unique_values:
                if value in value_counts:
                    value_counts[value] += 1
                else:
                    value_counts[value] = 1

    # Write the report to a CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Concept", "Num_Occurences"])
        for value, count in value_counts.items():
            writer.writerow([value, count])

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python count-concept-occurences.py <courses.json> <value_count_report.csv>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    generate_value_report(input_file, output_file)
    print(f"Value report saved to '{output_file}' in CSV format.")
