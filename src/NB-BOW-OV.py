import csv
import math

class NB_BOW_OV:

    '''
    Vocabulary is a dictionary which contains the number of times each word appears in factual and non-factual self.tweets.
    '''
    def __init__(self, smoothing=0.01):
        self.classes = ["yes", "no"]    # Classes
        self.vocab = {}                 # Dictionary contains {word : {self.classes[0] : num, self.classes[1] : num}} (e.g. {"this" : {"yes" : 1, "no" : 0}})
        self.tweets = {}                    # Dictionary containing {self.tweets : factual} (e.g. {"this is a tweet" : "no"})
        self.smoothing = smoothing 
        self.total_in_class = {}
        self.conditional_prob = {}
        self.prob_class = {}   


    '''
    Train the model with the given training set.
    '''
    def train(self, training_set_name):
        self.readFile(training_set_name)
        self.constructVocabulary()
        self.getTotalInClass()
        self.getConditionalProb()


    '''
    Read the training file and get the self.tweets and their factuality
    '''
    def readFile(self, filename):
        f = open(filename, "r", encoding="utf-8")
        reader = csv.reader(f, delimiter="\t")

        rows = list(reader)[1:]                 # We don't want the header for the training set
        twt = [r[1] for r in rows]              # Get the actual self.tweets
        factual = [r[2] for r in rows]          # Get whether the tweet is factual or not
        self.tweets = dict(zip(twt, factual))   # Construct the tweet dictionary
        
        self.getProbClass(factual)              # Get the probability of each class


    '''
    Construct the unfiltered vocabulary from the training set
    '''
    def constructVocabulary(self):
        # Go through each tweet and their factuality in the training set
        for twt, fact in self.tweets.items():
            splitTwt = list(twt.split(" "))        # Split the tweet into individual words

            # Go through each word in the tweet
            for word in splitTwt: 
                word = word.strip(".,?!@#:\"“-—\'()").lower()      # Strip the words of punctuation and set them to lowercase

                # Add the word to the vocabulary if it's not already in it
                if not word in self.vocab:
                    self.vocab[word] = {self.classes[0] : 0, self.classes[1] : 0}

                # Keep count of the number of times the word appears in factual/non-factual self.tweets
                if fact == self.classes[0]:
                    self.vocab[word][self.classes[0]] += 1
                elif fact == self.classes[1]:
                    self.vocab[word][self.classes[1]] += 1
        
        self.getTotalInClass()

    
    '''
    Get the total amount of words for each class
    '''
    def getTotalInClass(self):
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
            self.prob_class[c] = factual.count(c)/len(self.tweets)


    '''
    Get the conditional probability of each word given a class
    '''
    def getConditionalProb(self):
        for word in self.vocab:
            self.conditional_prob[word] = {self.classes[0] : float, self.classes[1] : float}
            for c in self.classes:
                self.conditional_prob[word][c] = (self.vocab[word][c] + self.smoothing)/(self.total_in_class[c] + (len(self.vocab) * self.smoothing))

    '''
    TODO
    '''
    def predict(self, test_set):
        return





    
nb = NB_BOW_OV()
nb.train("training/covid_training.tsv")
print(nb.conditional_prob)

    

    