import ft_classifier as classifier
from pprint import pprint

hyperparameters = {
    'minCount': 5,
    'dim': 50
}

classifier.train('training_slots.csv', hyperparameters=hyperparameters)
result = classifier.predict('how is the weather outside?')
classifier.save_vectors()
pprint(result)

pprint(classifier.word_freq(output_format='dict'))
