from _collections import defaultdict
from collections import Counter
from nltk.corpus import wordnet as wn
import numpy
from numpy.core.multiarray import ndarray, zeros
from numpy.random.mtrand import np
import re
from scipy.sparse.csgraph._min_spanning_tree import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
import textstat
import wikipedia
from textstat.textstat import textstat as ts


VALID_POS = ['FW', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNS', 
             'NNP', 'NNPS', 'PDT', 'POS', 'RB', 'RBR', 'RBS',
             'RP', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
             'WDT', 'WRB']

class Featurizer:
    def __init__(self, category_dict, features=defaultdict(lambda: False), use_default=False, analyzer=u'word'):
        self.stop_words=['uner', 'via', 'answer', 'ANSWER', 'was', 'his','the','that','and', 'points', 'pointsname', 'this', 'for', 'who', 'they','can', 'its', 'also', 'these', 'are', 'name', 'after', 'than', 'had', 'with', 'about', 'ftp', '10', '15', 'this', 'from', 'not', 'has', 'well']
        self.analyzer = analyzer
        self.category_dict = category_dict
        self.features = defaultdict(lambda: False)
        for feat in features:
            self.features[feat] = features[feat]
        
        if use_default:
            self.vectorizer = CountVectorizer(ngram_range=(1,10), stop_words=self.stop_words)
            self.default = True
        else:
            self.vectorizer = CountVectorizer(analyzer=analyzer)
            self.default = False
            
    def get_answer_features(self, features, answer, answer_wiki_features, provided_answer):
        annotated_words = ""
        question_string = ""
        if features['wiki_answer'] or features['wiki_num_results']:
            if answer in answer_wiki_features:
                num_results = answer_wiki_features[answer][0]
#                 print 'Num Results: '+str(num_results)
                first_len = answer_wiki_features[answer][1]
#                 print 'First len: '+str(first_len)
                num_occurences = answer_wiki_features[answer][2]
                question_string += "WIKI_NUM:"+str(num_results)+":WIKI_FIRST:"+str(first_len)+":"+str(num_occurences)
#                 print question_string
            else:
                question_string = ":0::0:0"
        else:
            question_string = ":0::0:0"
        annotated_words += question_string+":"
        return annotated_words

    
    def get_example_features(self, example_id, features, category_dict, question, user, answer_wiki_features, examples_list, default=False):
        current_category = question.category
        question_features = self.get_question_features(features, question, category_dict)
        user_features = self.get_user_features(features, user, current_category)
        answer = question.answer
        question_id = question.q_id
        user_questions = user.questions
        example_features = ""
        if features['previous_prediction']:
            example_features += str(float(examples_list[example_id].previous_prediction)) +":"
        if question_id in user_questions:
            provided_answer = user.questions[question.q_id]['answer']
        else:
            provided_answer = ""
        answer_features = self.get_answer_features(features, answer, answer_wiki_features, provided_answer)
        category_features = self.get_category_features(features, current_category, category_dict)
        if default:
            feature_list = []
            question_feature_list = question_features.split()
            for feature in question_feature_list:
                individual_features = feature.split(":")
                for current_individual_feature in individual_features:
                    feature_list.append(current_individual_feature)
            user_features_list = user_features.split()
            for feature in user_features_list:
                individual_features = feature.split(":")
                for current_individual_feature in individual_features:
                    feature_list.append(current_individual_feature)
    #         answer_features_list = answer_features
    #         for feature in answer_
            feature_string = " ".join(feature_list)
        else:
            feature_string = str(example_id)+"#"+user_features +"|"+question_features+"|"+answer_features+"|"+category_features+"|"+example_features
        return feature_string

    def train_feature(self, examples, users, wiki_data, skip_vectorizer=False, limit=-1):
        feature_names = {}
        index_count = 0
        if skip_vectorizer:
            thing_list = []
            for ex in self.all_examples(limit, examples, users, self.features, self.category_dict, wiki_data, default = self.default):
                for thing in self.analyzer.call_function(ex):
                    thing_list.append(thing)
#             count_features = csr_matrix([]);
            new_list = []
            for i in range(len(examples)):
                new_list.append(numpy.array([]))
            print "Numeric feature length: "+str(len(self.analyzer.numeric_features))
            matrix,numeric_features = self.analyzer.add_numeric_features(new_list)
            for feat in numeric_features:
                feature_names[index_count] = feat
                index_count += 1
            return matrix, feature_names
        else:
            count_features = self.vectorizer.fit_transform(ex for ex in self.all_examples(limit, examples, users, self.features, self.category_dict, wiki_data, default = self.default))
            count_features_names = self.vectorizer.vocabulary_
            for feat_name in count_features_names:
                feat_index = int(count_features_names[feat_name])
                feature_names[feat_index] = feat_name
                if feat_index > index_count:
                    index_count = feat_index
        if not self.default:
            matrix,numeric_features = self.analyzer.add_numeric_features(count_features.toarray())
            index_count += 1
            for feat in numeric_features:
                feature_names[index_count] = feat
                index_count += 1
            return matrix, feature_names
        return count_features, {}
            

    def test_feature(self, examples, users, wiki_data, skip_vectorizer=False, limit=-1):
        feature_names = {}
        index_count = 0
        if skip_vectorizer:
            thing_list = []
            for ex in self.all_examples(limit, examples, users, self.features, self.category_dict, wiki_data, default = self.default):
                for thing in self.analyzer.call_function(ex):
                    thing_list.append(thing)
#             count_features = csr_matrix(zeros(len(self.analyzer.numeric_features), 1).toarray());
            print "Numeric feature length: "+str(len(self.analyzer.numeric_features))
            new_list = []
            for i in range(len(examples)):
                new_list.append(numpy.array([]))
            matrix,numeric_features = self.analyzer.add_numeric_features(new_list)
            for feat in numeric_features:
                feature_names[index_count] = feat
            return matrix, feature_names
        else:
            count_features = self.vectorizer.transform(ex for ex in self.all_examples(limit, examples, users, self.features, self.category_dict, wiki_data, default = self.default))
            count_features_names = self.vectorizer.vocabulary_
            for feat_name in count_features_names:
                feat_index = int(count_features_names[feat_name])
                feature_names[feat_index] = feat_name
                if feat_index > index_count:
                    index_count = feat_index
        if not self.default:
            matrix,numeric_features = self.analyzer.add_numeric_features(count_features.toarray())
            index_count += 1
            for feat in numeric_features:
                feature_names[index_count] = feat
            return matrix, feature_names
        return count_features

    def show_topN(self, classifier, categories, N=10):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        for i, category in enumerate(categories):
            if(len(classifier.coef_) - 1 >= i):
                current = np.argsort(classifier.coef_[i])
                if len(current) > N:
                    topN = current[-N:]
                else:
                    topN = current
                print("%s: %s" % (category, " ".join(feature_names[topN])))
            else:
                print 'Could not classify ' + str(category)
                
    def show_top10(self, classifier, categories):
        self.show_topN(classifier, categories, 10)
            
    def show_top20(self, classifier, categories):
        self.show_topN(classifier, categories, 20)

    def all_examples(self, limit, examples, users, features, category_dict, answer_wiki_features, default=False):
        for i in range(len(examples)):
            current_example = examples[i]
            current_user = users[current_example.user]
            tagged_example_str = self.get_example_features(i,features, category_dict, current_example.question, current_user, answer_wiki_features, examples, default) 
            yield tagged_example_str


    def get_user_features(self, features, user, category):
        
        average = str(abs(user.average_position))
        user_features = ""
        if features['user_average']:
            user_features += average 
        user_features += ":"
        if features['user_category_average']:
            category_data = user.category_averages[category]
            user_features += str(float(category_data[0]))
        user_features += ":"
        if features['user_num_answered']:
            user_features += str(user.num_questions)
        user_features += ":"
        if features['user_num_incorrect']:
            user_features += str(user.num_incorrect)
        user_features += ":"
        if features['user_incorrect_average']:
            user_features += str(user.incorrect_average)
        user_features += ":"
        if features['user_correct_average']:
            user_features += str(user.correct_average)
            
        return user_features


    def get_category_features(self, pred_features, category, category_dict):
        category_string = ""
        if pred_features['category_average']:
            category_string += str(category_dict[category].average)
        return category_string
        
    def get_question_features(self, features, question_container, categories):
        question = question_container.question
        category = question_container.category
        keywords = question_container.keywords
        answer = question_container.answer
        question_id = question_container.q_id
        
        annotated_words = []
        stop_words=['uner', 'via', 'answer', 'ANSWER', 'was', 'his','the','that','and', 'points', 'pointsname', 'this', 'for', 'who', 'they','can', 'its', 'also', 'these', 'are', 'name', 'after', 'than', 'had', 'with', 'about', 'ftp', '10', '15', 'this', 'from', 'not', 'has', 'well']
        prev_word = ""
        prev_word_unannotated = ""
        reg_exp = re.compile(u'(?u)\\b[a-zA-Z]{2}[a-zA-Z]+|[0-9]{4}\\b')
        reg_exp_alphanum = re.compile(u'\\w')
    
        words = {}
        pos = {}
        
        question_mark_count = 0
        period_count = 0
        comma_count = 0
        double_quote_count = 0
        single_quote_count = 0
        asterisk_count = 0
        
        raw_words = []
        
        word_index = 0
        for word_tuple in question:
            word = word_tuple[0]
            raw_words.append(word)
            words[word_index] = word
            question_mark_count += word.count("?")
            period_count += word.count(".")
            comma_count += word.count(",")
            double_quote_count += word.count("\"")
            single_quote_count += word.count("'")
            asterisk_count += word.count("*")
            pos[word_index] = word_tuple[1]
            word_index = word_index + 1
            
        raw_question = " ".join(raw_words)
        syllables = ts.syllable_count(raw_question)
        sentences = ts.sentence_count(raw_question)
        grade_level = ts.flesch_reading_ease(raw_question)#ts.readability_consensus(raw_question)
        
        numeric_question_features = ""
        
        c = Counter([values for values in words.itervalues()])
        counts = dict(c)
        
        encounteredNoun = False
            
        for word_index in words:
            word = words[word_index]
            result = reg_exp.match(word)
            if result is None:
                continue
            original_annotated_word = result.string[result.start(0):result.end(0)]
            annotated_word = ""
            if features['word']:
                annotated_word = original_annotated_word
            if original_annotated_word.lower() in stop_words:
                prev_word_unannotated = word
                continue
            annotated_word += ":"
            if features['speech'] and word_index in pos:
                if pos[word_index] in VALID_POS:
                    annotated_word += pos[word_index]
                    if pos[word_index] in ['NN', 'NNS', 'NNP', 'NNPS']:
                        encounteredNoun = True
            annotated_word += ":"
            if features['capital'] and len(original_annotated_word) > 0 and original_annotated_word[0].isupper() and len(prev_word_unannotated) > 0 and reg_exp_alphanum.match(prev_word_unannotated[len(prev_word_unannotated)-1]):
                annotated_word+="C"
            annotated_word += ":"
            if features['all_upper'] and original_annotated_word.isupper():
                annotated_word+="U"
            annotated_word += ":"
            if features['foreign'] and not wn.synsets(original_annotated_word):
                annotated_word+="F"
            annotated_word += ":"
            
    
            if features['unique'] and word in counts:
                if counts[word] == 1:
                    annotated_word+="UN"
            annotated_word += ":"        
            #count number of words that are numeric
            if features['numbers']  and original_annotated_word.isdigit():
                annotated_word+="NUM"
            annotated_word += ":"
            if features['before_noun'] and not encounteredNoun:
                annotated_word+="BFRN"
        
            annotated_words.append(annotated_word)
            prev_word = original_annotated_word
            prev_word_unannotated = word
        
        tagged = " ".join(annotated_words)
        
        #####Numeric question features - seperated by a "_" symbol
        tagged += "_"
        if features['question_count']:
            tagged += str(int(question_container.count))
        tagged += ":"
        if features['question_average']:
            if question_container.average_response == 0:
                question_container.average_response = -abs(categories[category].average)
            tagged += str(question_container.average_response)
        tagged += ":"
        if features['question_percent']:
            tagged += str(question_container.absolute_average)
        tagged += ":"
        if features['question_answer_percent']:
            tagged += str(question_container.average_response_percent)
        tagged += ":"
        if features['question_length']:
            tagged += str(len(words))
        tagged += ":"
        if self.features['question_mark']:
            tagged += str(question_mark_count)
        tagged += ":"
        if self.features['question_sentence_count']:
            tagged += str(sentences)
        tagged += ":"
        if self.features['question_comma_count']:
            tagged += str(comma_count)
        tagged += ":"
        if self.features['question_double_quote_count']:
            tagged += str(double_quote_count)
        tagged += ":"
        if self.features['question_single_quote_count']:
            tagged += str(single_quote_count)
        tagged += ":"
        if self.features['question_asterisk_count']:
            tagged += str(asterisk_count)
        tagged += ":"
        if self.features['syllable_count']:
            tagged += str(syllables)
        tagged += ":"
        if self.features['grade_level']:
            tagged += str(grade_level)
        tagged += ":"
        if self.features['period_count']:
            tagged += str(period_count)
        tagged += ":"
        if self.features['question_incorrect_average']:
            tagged += str(question_container.incorrect_average)
        tagged += ":"
        if self.features['question_correct_average']:
            tagged += str(question_container.correct_average)
        tagged += ":"
        if self.features['question_length_average_difference']:
            tagged += str(int(len(question_container.question) - question_container.correct_average))
        return tagged