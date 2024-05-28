import os
from sklearn.ensemble import RandomForestClassifier

os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Ova e primerok od podatochnoto mnozestvo, za treniranje/evaluacija koristete ja
# importiranata promenliva dataset
dataset_sample = [[180.0, 23.6, 25.2, 27.9, 25.4, 14.0, 'Roach'],
                  [12.2, 11.5, 12.2, 13.4, 15.6, 10.4, 'Smelt'],
                  [135.0, 20.0, 22.0, 23.5, 25.0, 15.0, 'Perch'],
                  [1600.0, 56.0, 60.0, 64.0, 15.0, 9.6, 'Pike'],
                  [120.0, 20.0, 22.0, 23.5, 26.0, 14.5, 'Perch']]

if __name__ == '__main__':
    # Vashiot kod tuka
    col=int(input())

    train_set=dataset_sample[:int(len(dataset_sample)*0.85)]
    X_train=[[row[i] for i in range(len(row)) if i!=col] for row in train_set]
    y_train=[row[-1] for row in train_set]
    X_train=[row[:-1] for row in X_train]

    test_set=dataset_sample[int(len(dataset_sample)*0.85):]
    X_test=[[row[i] for i in range(len(row)) if i!=col] for row in test_set]
    y_test=[row[-1] for row in test_set]
    X_test=[row[:-1] for row in X_test]

    n_estimators = int(input())
    criteria = input()
    classifier=RandomForestClassifier(n_estimators=n_estimators, criterion=criteria, random_state=0)
    classifier.fit(X_train, y_train)

    acc=0
    for i in range(len(test_set)):
        predicted_class=classifier.predict([X_test[i]])[0]
        if predicted_class==y_test[i]:
            acc+=1
    inp=input().split(" ")
    inp= list(float(num) for num in inp)
    inp.pop(col)
    print(f"Accuracy: {acc/len(test_set)}")
    print(classifier.predict([inp])[0])
    print(classifier.predict_proba([inp])[0])


    # Na kraj potrebno e da napravite submit na podatochnoto mnozestvo
    # i klasifikatorot so povik na slednite funkcii

    # submit na trenirachkoto mnozestvo
    # submit_train_data(train_X, train_Y)

    # submit na testirachkoto mnozestvo
    # submit_test_data(test_X, test_Y)

    # submit na klasifikatorot
    # submit_classifier(classifier)
