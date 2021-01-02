import ft_classifier as classifier

classifier.train('training_slots.csv')
result = classifier.predict('how is the weather outside?')
print(result)

