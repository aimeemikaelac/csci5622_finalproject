from _collections import defaultdict
from scipy.sparse.csgraph._min_spanning_tree import csr_matrix


class Analyzer:
    def __init__(self, features):
        self.features = defaultdict(lambda: False)
        for feature in features:
            self.features[feature] = features[feature]
#         print self.features
        self.numeric_features = defaultdict(dict)
        self.example_index = 0
        self.example_index_table = {}
        self.word_counts = defaultdict(lambda: 0.0)
        
    
    def strip_tags_from_words(self, word_list):
        stripped_words = []
        for word_feature in word_list:
            stripped_words.append(word_feature.split(":")[0])
        return stripped_words
    
    def store_numeric_feature(self, example_index, feat_name, numeric_feature):
        self.numeric_features[example_index][feat_name] = float(numeric_feature)
        
    def process_question_string(self, feat_id, question_string):
#         question_features_array_ = question_string.split()
#         print question_string
        question_features_array = question_string.split("_")
        question_features = question_features_array[0].split()
        numeric_question_features = question_features_array[1]
        if self.features['ngram_range'] != (1,1) and self.features['ngram_range']!=False:
            current_words = self.strip_tags_from_words(question_features)
            for ngram_length in range(self.features['ngram_range'][0], self.features['ngram_range'][1]+1):
                    for ngram in zip(*[current_words[i:] for i in range(3)]):
                        ngram = " ".join(ngram)
                        yield ngram
        
        for feat_tuple in question_features:
            feat_tokens = feat_tuple.split(":")
            if self.features['word']: #and self.ngram_range == (1,1):
                pos = feat_tokens[1]
                if type(self.features['word_pos']) is list and pos in self.features['word_pos']:
                    word = feat_tokens[0]
                    self.word_counts[word] += 1
#                     self.store_numeric_feature(feat_id, word, self.word_counts[word])
#                     yield feat_tokens[0]
                
            if self.features['speech']:
                if feat_tokens[1] != "Unk":
                    yield feat_tokens[1]
                    
            if self.features['capital']:
                if feat_tokens[2] == "C":
                    yield feat_tokens[2]
                    
            if self.features['all_upper']:
                if feat_tokens[3] == "U":
                    yield feat_tokens[3]
                    
            if self.features['foreign']:
                if feat_tokens[4] == "F":
                    yield feat_tokens[4]
                    
            if self.features['unique']:
                if feat_tokens[5] == "UN":
                    yield feat_tokens[5]
                    
            if self.features['numbers']:
                if feat_tokens[6] == "NUM":
                    yield feat_tokens[6]
                    
            if self.features['before_noun']:
                if feat_tokens[7] == "BFRN":
                    yield feat_tokens[7]
                    
            if self.features['use_dictionary']:
                for category in self.dictionary:
                    if feat_tokens[0] in self.dictionary[category]:
                        yield category.upper()
        
        feat_tokens = numeric_question_features.split(":")
#         print feat_tokens
        if self.features['question_count']:
            self.store_numeric_feature(feat_id, 'question_count', int(feat_tokens[0]))
        
        if self.features['question_average']:
            self.store_numeric_feature(feat_id, 'question_average', float(feat_tokens[1]))
            
        if self.features['question_percent']:
            self.store_numeric_feature(feat_id, 'question_percent', float(feat_tokens[2]))
            
        if self.features['question_answer_percent']:
            self.store_numeric_feature(feat_id, 'question_answer_percent', float(feat_tokens[3]))
#                 print float(feat_tokens[11])
        
        if self.features['question_length']:
            self.store_numeric_feature(feat_id, 'question_length', int(feat_tokens[4]))
            
        if self.features['question_mark']:
            self.store_numeric_feature(feat_id, 'question_mark', int(feat_tokens[5]))
            
        if self.features['question_sentence_count']:
            self.store_numeric_feature(feat_id, 'question_sentence_count', int(feat_tokens[6]))
            
        if self.features['question_comma_count']:
            self.store_numeric_feature(feat_id, 'question_comma_count', int(feat_tokens[7]))
            
        if self.features['question_double_quote_count']:
            self.store_numeric_feature(feat_id, 'question_double_quote_count', int(feat_tokens[8]))
            
        if self.features['question_single_quote_count']:
            self.store_numeric_feature(feat_id, 'question_single_quote_count', int(feat_tokens[9]))
            
        if self.features['question_asterisk_count']:
            self.store_numeric_feature(feat_id, 'question_asterisk_count', int(feat_tokens[10]))
            
        if self.features['syllable_count']:
            self.store_numeric_feature(feat_id, 'syllable_count', float(feat_tokens[11]))
            
        if self.features['grade_level']:
            self.store_numeric_feature(feat_id, 'grade_level', float(feat_tokens[12]))
            
        if self.features['period_count']:
            self.store_numeric_feature(feat_id, 'syllable_count', int(feat_tokens[13]))
            
        if self.features['question_incorrect_average']:
            self.store_numeric_feature(feat_id, 'question_incorrect_average', float(feat_tokens[14]))
            
        if self.features['question_correct_average']:
            self.store_numeric_feature(feat_id, 'question_correct_average', float(feat_tokens[15]))
            
        if self.features['question_length_average_difference']:
            self.store_numeric_feature(feat_id, 'question_length_average_difference', float(feat_tokens[16]))
            
                        
    def process_user_string(self,feat_id, user_string):
        user_features_tokens = user_string.split(":")
        if self.features['user_average']:
            self.store_numeric_feature(feat_id, 'user_average', float(user_features_tokens[0]))
        if self.features['user_category_average']:
#                 print float(user_features_tokens[3])
            self.store_numeric_feature(feat_id, 'user_category_average', float(user_features_tokens[1]))
        if self.features['user_num_answered']:
            self.store_numeric_feature(feat_id, 'user_num_answered', int(user_features_tokens[2]))
        if self.features['user_num_incorrect']:
            self.store_numeric_feature(feat_id, 'user_num_incorrect', int(user_features_tokens[3]))
        if self.features['user_incorrect_average']:
            self.store_numeric_feature(feat_id, 'user_incorrect_average', float(user_features_tokens[4]))
        if self.features['user_correct_average']:
            self.store_numeric_feature(feat_id, 'user_correct_average', float(user_features_tokens[5]))

    def process_answer_string(self, feat_id, answer_string):
        answer_features_tokens = answer_string.split(":")
        if self.features['wiki_num_results']:
#             print answer_features_tokens
            if answer_features_tokens[0] == "WIKI_NUM":
#                 print "WIKI_NUM"
                self.store_numeric_feature(feat_id, 'wiki_num', int(answer_features_tokens[1]))
        if self.features['wiki_answer']:
            if answer_features_tokens[2] == "WIKI_FIRST":
                self.store_numeric_feature(feat_id, 'wiki_first', int(answer_features_tokens[3]))
        if self.features['provided_answer']:
            self.store_numeric_feature(feat_id, 'provided_answer', int(answer_features_tokens[4]))

    def process_category_string(self, feat_id, category_string):
        category_features_tokens = category_string.split(":")
        if self.features['category_average']:
            self.store_numeric_feature(feat_id, 'category_average', float(category_features_tokens[0]))
            
    def process_example_string(self, feat_id, example_string):
        example_features_tokens = example_string.split(":")
        if self.features['previous_prediction']:
#             print "Processing example string: "+example_string
            self.store_numeric_feature(feat_id, 'previous_prediction', example_features_tokens[0])
    
    def add_numeric_features(self, feature_matrix_array):
#         feature_array = feature_matrix.toarray()
        feature_names = []
        found_features = False
        feature_array = feature_matrix_array
        new_feature_array = []
        for i in range(len(feature_array)):
            current_features = feature_array[i].tolist()
            for feat_index in self.numeric_features[i]:
                if not found_features:
                    feature_names.append(feat_index)
                current_features.append(self.numeric_features[i][feat_index])
            found_features = True
            new_feature_array.append(current_features)
        return csr_matrix(new_feature_array),feature_names
    
    def __call__(self, feature_string):
        for thing in self.call_function(feature_string):
            yield thing
        
    def call_function(self, feature_string):
        feature_strings_ = feature_string.split("#")
        example_id = int(feature_strings_[0])
        if example_id in self.example_index_table:
            feat_id = self.example_index_table[example_id]
        else:
            feat_id = self.example_index
            self.example_index += 1
            self.example_index_table[example_id] = feat_id
        feature_strings = feature_strings_[1].split("|")
        
        for question_feat in self.process_question_string(feat_id, feature_strings[1]):
            yield question_feat
        
        self.process_user_string(feat_id, feature_strings[0])
#         for user_feat in self.process_user_string(feat_id, feature_strings[0]):
#             yield user_feat
            
        self.process_answer_string(feat_id, feature_strings[2])    
#         for answer_feat in self.process_answer_string(feat_id, feature_strings[2]):
#             yield answer_feat
        self.process_category_string(feat_id, feature_strings[3])
#         for category_feat in self.process_category_string(feat_id, feature_strings[3]):
#             yield category_feat
        self.process_example_string(feat_id, feature_strings[4])