
from sklearn.model_selection import train_test_split, KFold
from user_generation import input_data, labels


dataset = input_data #users and articles [user, article_uri]
labels = labels #like or dislike [1/0]
i=0

X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)
print (X_train[0], y_train[0])
print (X_test[0], y_test[0])

kf = KFold(n_splits=10, random_state=42, shuffle=True)
for train_index, test_index in kf.split(dataset, labels):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = dataset[train_index[i]], dataset[test_index[i]]
    y_train, y_test = labels[train_index[i]], labels[test_index[i]]
    i+=1






