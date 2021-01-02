class intent_classifier:
    
    def __init__(self):
        self.model : NaiveBayesClassifier = None
        self.training_data = [] #list of tuples (training_phrase, class_label)
    
    def add_training_sample(self,sample):
        '''sample should be (text,class)'''
        if (len(sample) < 2):
            print("Error, sample format incorrect")
        else:
            self.training_data.append((sample[0],sample[1]))
    
    def train_model(self):
        '''using self.training_data, train self.model'''
        pass
        
    def classify_intent(self,text):
        '''Given the text (sentence from chat / a document), classify it'''
        
        #the class that the model predicts, from the given text
        predicted_class = None
        
        #number from 0 to 1, with prediction confidence or probability 
        probability = 0
        
        return (predicted_class,probability)


#global functions
def load_from_csv(classifier, filename):
    import csv
    with open(filename,newline="") as csvfile:
        all_rows = csv.reader(csvfile,delimiter=",")
        for row in all_rows:
            #print(row)
            classifier.add_training_sample(row)
            
    classifier.train_model()
    
def save_to_csv(classifier, filename):
    import csv
    with open(filename, "w",newline="") as csvfile:
        writer = csv.writer(csvfile,delimiter=",")
        writer.writerows(classifier.training_data)
    
    
def main():    
    new_classifier = intent_classifier()
    load_from_csv(new_classifier,".\\training_slots.csv")   
    
    new_classifier.train_model()
    
    print(new_classifier.training_data)
    
    prediction = new_classifier.classify_intent("I've just been reading today.")
     
if __name__ == "__main__":
    main()