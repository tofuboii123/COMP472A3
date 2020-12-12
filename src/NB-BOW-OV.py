import csv
import math
import operator
import numpy as np

class NB_BOW_OV:

    '''
    Vocabulary is a dictionary which contains the number of times each word appears in factual and non-factual self.tweets.
    '''
    def __init__(self, smoothing=0.01):
        self.classes = ["yes", "no"]        # Classes
        self.vocab = {}                     # Dictionary contains {word : {self.classes[0] : num, self.classes[1] : num}} (e.g. {"this" : {"yes" : 1, "no" : 0}})
        self.training_tweets = {}                    # Dictionary containing {self.tweets : factual} (e.g. {"this is a tweet" : "no"})
        self.smoothing = smoothing 
        self.total_in_class = {}
        self.conditional_prob = {}
        self.prob_class = {}
        self.test_tweets = {}   


    '''
    Train the model with the given training set.
    '''
    def train(self, training_set_name):
        self.readTrainingFile(training_set_name)
        self.constructVocabulary()
        self.setTotalInClass()
        self.setConditionalProb()


    '''
    Read the training file and get the self.tweets and their factuality
    '''
    def readTrainingFile(self, file_name):
        f = open(file_name, "r", encoding="utf-8")
        reader = csv.reader(f, delimiter="\t")

        rows = list(reader)[1:]                 # We don't want the header for the training set
        twt = [r[1] for r in rows]              # Get the actual self.tweets
        factual = [r[2] for r in rows]          # Get whether the tweet is factual or not
        self.training_tweets = dict(zip(twt, factual))   # Construct the tweet dictionary
        
        self.getProbClass(factual)              # Get the probability of each class


    '''
    Construct the unfiltered vocabulary from the training set
    '''
    def constructVocabulary(self):
        # Go through each tweet and their factuality in the training set
        for twt, fact in self.training_tweets.items():
            splitTwt = list(twt.split(" "))        # Split the tweet into individual words

            # Go through each word in the tweet
            for word in splitTwt: 
                word = word.lower()      # Strip the words of punctuation and set them to lowercase

                # Add the word to the vocabulary if it's not already in it
                if not word in self.vocab:
                    self.vocab[word] = {self.classes[0] : 0, self.classes[1] : 0}

                # Keep count of the number of times the word appears in factual/non-factual self.tweets
                if fact == self.classes[0]:
                    self.vocab[word][self.classes[0]] += 1
                elif fact == self.classes[1]:
                    self.vocab[word][self.classes[1]] += 1
        
        self.setTotalInClass()

    
    '''
    Get the total amount of words for each class
    '''
    def setTotalInClass(self):
        for c in self.classes:
            total = 0
            for _, fact in self.vocab.items():
                total += fact[c]
            
            self.total_in_class[c] = total

    '''
    Get the probabilities of each class
    '''
    def getProbClass(self, factual):
        for c in self.classes:
            self.prob_class[c] = factual.count(c)/len(self.training_tweets)


    '''
    Get the conditional probability of each word given a class
    '''
    def setConditionalProb(self):
        for word in self.vocab:
            self.conditional_prob[word] = {self.classes[0] : float, self.classes[1] : float}
            for c in self.classes:
                self.conditional_prob[word][c] = (self.vocab[word][c] + self.smoothing)/(self.total_in_class[c] + (len(self.vocab) * self.smoothing))


    '''
    Read in test files
    '''
    def readTestFile(self, file_name):
        f = open(file_name, "r", encoding="utf-8")
        reader = csv.reader(f, delimiter="\t")

        rows = list(reader)                             # We don't want the header for the training set
        ids = [r[0] for r in rows]                      # Tweet ids
        twt = [r[1] for r in rows]                      # Get the actual self.tweets
        factual = [r[2] for r in rows]                  # Get whether the tweet is factual or not

        # Get the values of the test set
        for i, num in enumerate(ids):
            self.test_tweets[num] = [twt[i], factual[i]]


    '''
    Get all words from document
    '''
    def getWords(self, document):
        words = []

        # Get each word and strip it
        for entry in document:
            splitTwt = list(entry.split(" "))
            for w in splitTwt:
                words.append(w.lower())

        return words


    '''
    Get the score of each class
    '''
    def getScore(self, document):
        score = {}                          # Class score
        prob_sum = {"yes" : 0, "no" : 0}    # Probability for each word in document

        # For each class
        for c in self.classes:
            score[c] = math.log10(self.prob_class[c])

            # Go through all the words in the document
            for word in self.getWords(document):

                # Only use words that are in the vocabulary (aka have a probability)
                if word in self.conditional_prob:
                    prob_sum[c] += math.log10(self.conditional_prob[word][c])

            score[c] += prob_sum[c]         # Score

        # Return the max score and its class
        return (score["yes"], "yes") if score["yes"] >= score["no"] else (score["no"], "no")

    
    '''
    Write trace file
    '''
    def writePredictions(self, trace_name):
        TP = 0
        FP = 0 
        FN = 0
        TN = 0
        totalCorrect = 0
        with open(trace_name, "w") as f:
            for num, entry in nb.test_tweets.items():
                score, prediction = self.getScore(entry)
                f.write("{}  {}  {:.2e}  {}  {}\n".format(num, prediction, score, entry[1], "correct" if prediction == entry[1] else "wrong"))
                
                if prediction == entry[1]:
                    totalCorrect += 1
                if prediction == "yes" and entry[1] == "yes":
                    TP += 1
                if prediction == "yes" and entry[1] == "no":
                    FP += 1
                if prediction == "no" and entry[1] == "yes":
                    FN += 1
                if prediction == "no" and entry[1] == "no":
                    TN += 1
        return TP, FP, FN, TN, totalCorrect


    '''
    Predict the results for each entry in the test set
    '''
    def predict(self, test_set, trace_name):
        self.readTestFile(test_set)
        TP, FP, FN, TN, totalCorrect = self.writePredictions(trace_name)


    '''
    Accuracy
    '''
    def accuracy(self, trace_name):
        TP, FP, FN, TN, totalCorrect = self.writePredictions(trace_name)
        totalWords = len(nb.test_tweets.items())
        accuracy = totalCorrect / totalWords
        return accuracy 


    '''
    Precision Calculation
    '''
    def precision(self, trace_name):
        TP, FP, FN, TN, totalCorrect = self.writePredictions(trace_name)
        precisionValueA = TP/(TP+FP)
        precisionValueB = TN/(TN+FN)
        return precisionValueA, precisionValueB
        
    '''
    Recall Calculation
    '''
    def recall(self, trace_name):
        TP, FP, FN, TN, totalCorrect = self.writePredictions(trace_name)
        recallValueA = TP/(TP+FN)
        recallValueB = TN/(TN+FP)
        return recallValueA, recallValueB
    
    '''
    FMeasure Calculation
    '''
    def F1Measure(self, trace_name):
        precision = self.precision(trace_name)
        recall = self.recall(trace_name)
        f1MeasureA = (2 * precision[0] * recall[0]) / (precision[0] + recall[0])
        f1MeasureB = (2 * precision[1] * recall[1]) / (precision[1] + recall[1])
        return f1MeasureA, f1MeasureB 
    
    def writeToText(self):
        precision1, precision2 = self.precision("./trace/trace_NB-BOW-OV.txt")
        recall1, recall2 = self.recall("./trace/trace_NB-BOW-OV.txt")
        f1Measure1, f1Measure2 = self.F1Measure("./trace/trace_NB-BOW-OV.txt")
        with open("./trace/eval_NB-BOW-OV.txt", "w") as file:
            file.write("{:.4}\n".format(self.accuracy("./trace/trace_NB-BOW-OV.txt")))
            file.write("{:.4}  {:.4}\n".format(precision1, precision2))
            file.write("{:.4}  {:.4}\n".format(recall1, recall2))
            file.write("{:.4}  {:.4}\n".format(f1Measure1, f1Measure2))
        file.close()
        
    
    
nb = NB_BOW_OV()
nb.train("./training/covid_training.tsv")
print(len(nb.vocab))

print(nb.total_in_class["no"])
print(nb.total_in_class["yes"])
print(nb.total_in_class["yes"] + nb.total_in_class["no"])
nb.predict("./test/covid_test_public.tsv", "./trace/trace_NB-BOW-OV.txt")
nb.writeToText()

# Test for metrics
# print(nb.accuracy("./trace/trace_NB-BOW-OV.txt"))
# print(nb.precision("./trace/trace_NB-BOW-OV.txt"))
# print(nb.recall("./trace/trace_NB-BOW-OV.txt"))
# print(nb.F1Measure("./trace/trace_NB-BOW-OV.txt"))
