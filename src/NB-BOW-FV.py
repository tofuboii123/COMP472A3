import csv
import math
import operator

class NB_BOW_FV:
    '''
    Vocabulary is a dictionary which contains the number of times each word appears in factual and non-factual self.tweets.
    '''
    def __init__(self, smoothing=0.01):
        self.classes = ["yes", "no"]        # Classes
        self.vocab = {}                     # Dictionary contains {word : {self.classes[0] : num, self.classes[1] : num}} (e.g. {"this" : {"yes" : 1, "no" : 0}})
        self.training_tweets = {}           # Dictionary containing {self.tweets : factual} (e.g. {"this is a tweet" : "no"})
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
       
        # Unfiltered vocabulary dictionary to keep count of each word's frequency and Filtered Vocabulary list 
        unfiltered_vocab = {}
        
        # Go through each tweet and their factuality in the training set
        for twt, fact in self.training_tweets.items():
            splitTwt = list(twt.split(" "))        # Split the tweet into individual words

            # Go through each word in the tweet
            for word in splitTwt: 
                word = word.strip(".,?!@#:\"“-—\'()").lower()                        # Strip the words of punctuation and set them to lowercase
                
                # Add the word to the vocabulary if it's not already in it
                if not word in self.vocab:
                    unfiltered_vocab[word] = {self.classes[0] : 0, self.classes[1] : 0}
                
                # Keep count of the number of times the word appears in factual/non-factual self.tweets
                if fact == self.classes[0]:
                    unfiltered_vocab[word][self.classes[0]] += 1
                elif fact == self.classes[1]:
                    unfiltered_vocab[word][self.classes[1]] += 1
                
        # Get all words with 2 or more factuals
        for word in unfiltered_vocab.keys(): 
            if((unfiltered_vocab[word][self.classes[0]] + unfiltered_vocab[word][self.classes[1]]) >= 2):
                self.vocab[word][self.classes[0]] = unfiltered_vocab[word][self.classes[0]]
                self.vocab[word][self.classes[0]] = unfiltered_vocab[word][self.classes[1]]

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
                words.append(w.strip(".,?!@#:\"“-—\'()").lower())

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
        with open(trace_name, "w") as f:
            for num, entry in nb.test_tweets.items():
                score, prediction = self.getScore(entry)
                f.write("{}  {}  {:.2e}  {}  {}\n".format(num, prediction, score, entry[1], "correct" if prediction == entry[1] else "wrong"))



    '''
    Predict the results for each entry in the test set
    '''
    def predict(self, test_set, trace_name):
        self.readTestFile(test_set)
        self.writePredictions(trace_name)
        

    
nb = NB_BOW_FV()
nb.train("training/covid_training.tsv")
nb.predict("test/covid_test_public.tsv", "trace/trace_NB-BOW-OV.txt")

