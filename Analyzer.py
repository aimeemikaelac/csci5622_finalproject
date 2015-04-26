from _collections import defaultdict
from scipy.sparse.csgraph._min_spanning_tree import csr_matrix


class Analyzer:
    def __init__(self, features):
        self.features = defaultdict(lambda: False)
        for feature in features:
            self.features[feature] = features[feature]
        print self.features
        self.numeric_features = defaultdict(dict)
        self.index = 0
    
    def strip_tags_from_words(self, word_list):
        stripped_words = []
        for word_feature in word_list:
            stripped_words.append(word_feature.split(":")[0])
        return stripped_words
    
    def store_numeric_feature(self, example_index, feat_name, numeric_feature):
        self.numeric_features[example_index][feat_name] = float(numeric_feature)
        
    def process_question_string(self, feat_id, question_string):
        question_features = question_string.split()
        if self.features['ngram_range'] != (1,1):
            current_words = self.strip_tags_from_words(question_features)
            for ngram_length in range(self.features['ngram_range'][0], self.features['ngram_range'][1]+1):
                    for ngram in zip(*[current_words[i:] for i in range(3)]):
                        ngram = " ".join(ngram)
                        yield ngram
        
        for feat_tuple in question_features:
            feat_tokens = feat_tuple.split(":")
            if self.features['word']: #and self.ngram_range == (1,1):
                yield feat_tokens[0]
                
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
                    
            if self.features['question_count']:
                self.store_numeric_feature(feat_id, 'question_count', int(feat_tokens[8]))
            
            if self.features['question_average']:
                self.store_numeric_feature(feat_id, 'question_average', float(feat_tokens[9]))
                
            if self.features['question_percent']:
                self.store_numeric_feature(feat_id, 'question_percent', float(feat_tokens[10]))
                
            if self.features['question_answer_percent']:
                self.store_numeric_feature(feat_id, 'question_answer_percent', float(feat_tokens[11]))
#                 print float(feat_tokens[11])
            
            if self.features['question_length']:
                self.store_numeric_feature(feat_id, 'question_length', int(feat_tokens[12]))
                
                    
            if self.features['use_dictionary']:
                for category in self.dictionary:
                    if feat_tokens[0] in self.dictionary[category]:
                        yield category.upper()
                        
    def process_user_string(self,feat_id, user_string):
        user_features_tokens = user_string.split(":")
        if self.features['user_average']:
            self.store_numeric_feature(feat_id, 'user_average', float(user_features_tokens[0]))
        if self.features['user_category_average']:
#                 print float(user_features_tokens[3])
            self.store_numeric_feature(feat_id, 'user_category_average', float(user_features_tokens[1]))

    def process_answer_string(self, feat_id, answer_string):
        answer_features_tokens = answer_string.split(":")
        if self.features['wiki_answer']:
            if answer_features_tokens[2] == "WIKI_FIRST":
                self.store_numeric_feature(feat_id, 'wiki_first', int(answer_features_tokens[3]))
        if self.features['provided_answer']:
            self.store_numeric_feature(feat_id, 'provided_answer', int(answer_features_tokens[4]))

    def process_category_string(self, feat_id, category_string):
        category_features_tokens = category_string.split(":")
        if self.features['category_average']:
            self.store_numeric_feature(feat_id, 'category_average', float(category_features_tokens[0]))
    
    def add_numeric_features(self, feature_matrix):
        feature_array = feature_matrix.toarray()
        new_feature_array = []
        for i in range(len(feature_array)):
            current_features = feature_array[i].tolist()
            for feat_index in self.numeric_features[i]:
                current_features.append(self.numeric_features[i][feat_index])
            new_feature_array.append(current_features)
        return csr_matrix(new_feature_array)
    
    def __call__(self, feature_string):
        feature_strings_ = feature_string.split("#")
        feat_id = int(feature_strings_[0])
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