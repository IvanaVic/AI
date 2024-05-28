import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
# from submission_script import *
# from dataset_script import dataset

# Ova e primerok od podatochnoto mnozestvo, za treniranje/evaluacija koristete ja
# importiranata promenliva dataset
dataset_sample = [['C', 'S', 'O', '1', '2', '1', '1', '2', '1', '2', '0'],
                  ['D', 'S', 'O', '1', '3', '1', '1', '2', '1', '2', '0'],
                  ['C', 'S', 'O', '1', '3', '1', '1', '2', '1', '1', '0'],
                  ['D', 'S', 'O', '1', '3', '1', '1', '2', '1', '2', '0'],
                  ['D', 'A', 'O', '1', '3', '1', '1', '2', '1', '2', '0']]

if __name__ == '__main__':
    # Vashiot kod tuka
    encoder = OrdinalEncoder()
    encoder.fit([row[:-1] for row in dataset_sample])

    train_set=dataset_sample[:int(len(dataset_sample)*0.75)]
    train_X=[row[:-1] for row in train_set]
    train_Y=[row[-1] for row in train_set]
    train_X = encoder.transform(train_X)

    test_set=dataset_sample[int(len(dataset_sample)*0.75):]
    test_X = [row[:-1] for row in test_set]
    test_Y = [row[-1] for row in test_set]
    test_X = encoder.transform(test_X)

    classifier=CategoricalNB()
    classifier.fit(train_X, train_Y)

    acctual_class=0

    for i in range(len(test_X)):
        predicted_class=classifier.predict([test_X[i]])[0]
        if predicted_class==test_Y[i]:
            acctual_class+=1
    print(acctual_class/len(test_X))
    i=encoder.transform([input().split(' ')])
    print(classifier.predict(i)[0])
    print(classifier.predict_proba(i))





    # Na kraj potrebno e da napravite submit na podatochnoto mnozestvo,
    # klasifikatorot i encoderot so povik na slednite funkcii

    # submit na trenirachkoto mnozestvo
    # submit_train_data(train_X, train_Y)

    # submit na testirachkoto mnozestvo
    # submit_test_data(test_X, test_Y)

    # submit na klasifikatorot
    # submit_classifier(classifier)

    # submit na encoderot
    # submit_encoder(encoder)

    # povtoren import na kraj / ne ja otstranuvajte ovaa linija
    # from submission_script import *
