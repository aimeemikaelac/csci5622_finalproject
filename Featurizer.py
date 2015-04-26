from _collections import defaultdict
from collections import Counter
from numpy.random.mtrand import np
from nltk.corpus import wordnet as wn
import re
from sklearn.feature_extraction.text import CountVectorizer
import wikipedia


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
        question_string = "'"
        incorrect_threshold = 10
        if features['wiki_answer']:
            if answer in answer_wiki_features:
                num_results = answer_wiki_features[answer][0]
                first_len = answer_wiki_features[answer][1]
                num_occurences = answer_wiki_features[answer][2]
                question_string += "WIKI_NUM:"+str(num_results)+":WIKI_FIRST:"+str(first_len)+":"+str(num_occurences)
            else:
                try:
                    results, suggestion = eval(str(wikipedia.search(answer, results=1000000, suggestion=True)))
                    if suggestion is not None:
                        print "Suggestion: "+str(suggestion)+" "+str(len(results))+" answer: "+answer
                        if len(results) <= incorrect_threshold:
                            results = eval(str(wikipedia.search(str(suggestion), results=1000000)))
                    num_results = len(results)
                    question_string += "WIKI_NUM:"+str(num_results)+":WIKI_FIRST:"
                    if num_results > 0:
                        first_result = wikipedia.page(str(results[0]))
                        first_len = len(first_result.content)
                        question_string+=str(first_len)
                        
                        if features['provided_answer'] and len(provided_answer) > 0:
                            num_occurences = results.lower().count(provided_answer.lower())
                            print num_occurences
                            question_string += ":"+str(num_occurences)
                        else:
                            num_occurences = 0
                            question_string += ":0"
                        answer_wiki_features[answer] = [num_results,first_len, num_occurences, results, first_result]
                    else:
                        question_string+="0:0"
                        answer_wiki_features[answer] = [num_results, 0, 0, results, None]
                except:
                    question_string = "WIKI_NUM:0:WIKI_FIRST:0:0"
                    answer_wiki_features[answer] = [0,0, 0, None, None]
        else:
            question_string = "::::0"
        annotated_words += question_string+":"
        return annotated_words

    
    def get_example_features(self, example_id, features, category_dict, question, user, answer_wiki_features, default=False):
        current_category = question.category
        question_features = self.get_question_features(features, question)
        user_features = self.get_user_features(features, user, current_category)
        answer = question.answer
        question_id = question.q_id
        user_questions = user.questions
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
            feature_string = str(example_id)+"#"+user_features +"|"+question_features+"|"+answer_features+"|"+category_features
        return feature_string

    def train_feature(self, examples, users, wiki_data, limit=-1):
        count_features = self.vectorizer.fit_transform(ex for ex in self.all_examples(limit, examples, users, self.features, self.category_dict, wiki_data, default = self.default))
        if not self.default:
            return self.analyzer.add_numeric_features(count_features)
        return count_features
            

    def test_feature(self, examples, users, wiki_data, limit=-1):
        count_features = self.vectorizer.transform(ex for ex in self.all_examples(limit, examples, users, self.features, self.category_dict, wiki_data, default = self.default))
        if not self.default:
                return self.analyzer.add_numeric_features(count_features)
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
            tagged_example_str = self.get_example_features(i,features, category_dict, current_example.question, current_user, answer_wiki_features, default) 
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
            
        return user_features


    def get_category_features(self, pred_features, category, category_dict):
        category_string = ""
        if pred_features['category_average']:
            category_string += str(category_dict[category].average)
        return category_string
        
    def get_question_features(self, features, question_container):
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
        
        word_index = 0
        for word_tuple in question:
            words[word_index] = word_tuple[0]
            pos[word_index] = word_tuple[1]
            word_index = word_index + 1
        
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
            if features['capital'] and original_annotated_word[0].isupper() and len(prev_word_unannotated) > 0 and reg_exp_alphanum.match(prev_word_unannotated[len(prev_word_unannotated)-1]):
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
            annotated_word += ":"
            if features['question_count']:
                annotated_word += str(int(question_container.count))
            annotated_word += ":"
            if features['question_average']:
                annotated_word += str(question_container.average_response)
            annotated_word += ":"
            if features['question_percent']:
                annotated_word += str(question_container.absolute_average)
            annotated_word += ":"
            if features['question_answer_percent']:
                annotated_word += str(question_container.average_response_percent)
            annotated_word += ":"
            if features['question_length']:
                annotated_word += str(len(words))
            
            
            annotated_words.append(annotated_word)
            prev_word = original_annotated_word
            prev_word_unannotated = word
            
        tagged = " ".join(annotated_words)
        return tagged