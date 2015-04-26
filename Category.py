class Category:
    def __init__(self, category):
        self.category = category
        self.questions = []
        self.average = 0.0
        self.abs_total = 0.0
        self.total = 0.0
        self.count = 0
        self.users = []
        
    def add_question(self, q_id):
        self.questions.append(q_id)
        
    def add_occurrence(self, user_container, user_response):
        self.count += 1
        self.total += user_response
        self.abs_total += abs(user_response)
        self.average = float(self.total)/float(self.count)
        self.users.append(user_container)