import re

class Course():

    ID_PATTERN = r"C_course-v1:(\w+)\+(\w+)(?:\+\w+)?"

    def __init__(self, course_id, teacher_rank, school_rank, concepts, prerequisites):
        self.course_id = course_id
        self.concepts = concepts
        self.prerequisites = prerequisites
        self.teacher_rank = teacher_rank
        self.school_rank = school_rank
        #TODO add course popularity as dimension

    def __hash__(self):
        return hash(self.course_id)
    
    def __eq__(self, other):
        if not isinstance(other, Course):
            return False
        
        self_id = re.match(Course.ID_PATTERN, self.course_id)
        other_id = re.match(Course.ID_PATTERN, other.course_id)
        
        if not (self_id or other_id):
            return False
        
        return self_id.groups()[1] == other_id.groups()[1]
    
    def get_as_input(self):
        vector =  [self.teacher_rank, self.school_rank] + self.concepts
        return vector

    def get_as_target(self):
        vector =  [self.teacher_rank, self.school_rank] + self.prerequisites
        return vector