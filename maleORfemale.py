from sklearn import tree

#Weight, height, shoe size
X = [
    [190, 80, 45], [200 , 77, 42], [247, 77, 47], [250, 80, 50],
    [190, 70, 44], [ 210, 80, 47 ], [180, 74, 40], [177, 70, 40]
]

Y = ['male', 'female', 'male', 'male', 
    'female', 'male', 'female', 'female']

print("creating classifier and fitting model")
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(X, Y)


r1 = [210, 70, 47] #male
r2 = [210, 70, 42] #female
print("predicting...")
prediction = classifier.predict([r1])

print(prediction)