class Question:
    def __init__(self, question, category, keywords, answer, q_type, q_id):
        self.question = question
        self.category = category
        self.keywords = keywords
        self.answer = answer
        self.q_type = q_type
        self.q_id = int(q_id)
        self.num_correct = 0
        self.num_incorrect = 0
        self.average_response = 0.0
        self.absolute_average = 0.0
        self.percent_correct = 0
        self.count = 0
        self.min = 0.0
        self.running_total = 0.0
        self.running_magnitude_total = 0.0
        self.length = len(question)
        self.average_response_percent = 0.0
        
    def add_question_occurence(self, response_time):
        if self.count == 0 or self.min >= abs(response_time):
            self.min = abs(response_time)
        if response_time == 0:
            response_time = self.absolute_average
#         current_total = float(self.average_response) * float(self.count)
        self.running_total += response_time
        self.running_magnitude_total += abs(response_time)
        self.count += 1
        if response_time > 0:
            self.num_correct += 1
        else:
            self.num_incorrect += 1
            
        self.percent_correct = float(self.num_correct) / float(self.count)
        self.absolute_average = (float(self.running_total))/float(self.count)
        self.average_response = (float(self.running_magnitude_total))/float(self.count)
        self.average_response_percent = self.average_response/self.length
        if self.average_response_percent > 1:
            print "Cannot be greater than 100%"
            raise Exception(response_time, self.count, self.absolute_average, self.average_response, self.average_response_percent, self.length)