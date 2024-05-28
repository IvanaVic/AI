import os
from sklearn.naive_bayes import GaussianNB
os.environ['OPENBLAS_NUM_THREADS'] = '1'


# Ova e primerok od podatochnoto mnozestvo, za treniranje/evaluacija koristete ja
# importiranata promenliva dataset
dataset_sample = [['1', '35', '12', '5', '1', '100', '0'],
                  ['1', '29', '7', '5', '1', '96', '1'],
                  ['1', '50', '8', '1', '3', '132', '0'],
                  ['1', '32', '11.75', '7', '3', '750', '0'],
                  ['1', '67', '9.25', '1', '1', '42', '0']]

if __name__ == '__main__':

    data_sample=[]

    for row in dataset_sample:
        r = [float(el) for el in row[:-1]]
        r.append(int(row[-1]))
        data_sample.append(r)


    train_set=data_sample[:int(len(data_sample)*0.85)]
    X_train=[row[:-1] for row in train_set]
    Y_train=[row[-1] for row in train_set]

    test_set=data_sample[int(len(data_sample)*0.85):]
    X_test=[row[:-1] for row in test_set]
    Y_test=[row[-1] for row in test_set]

    classifier=GaussianNB()
    classifier.fit(X_train, Y_train)

    accual_class=0

    for i in range(len(X_test)):
        predicted_class=classifier.predict([X_test[i]])[0]
        if predicted_class==Y_test[i]:
            accual_class=accual_class+1

    print(accual_class/len(test_set))

    j=[float(num) for num in input().split(" ")]
    i=[]
    for num in i:
        i.append(float(num))


    print(classifier.predict([i])[0])
    print(classifier.predict_proba([i]))