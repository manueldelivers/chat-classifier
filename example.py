import ft_classifier as classifier

hyperparameters = {
    'minCount': 5,
    'dim': 20
}

classifier.train('training_slots.csv', hyperparameters=hyperparameters)
result = classifier.predict('how is the weather outside?')
print(result)

