import os
from scipy import misc
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

def input_images(directory):
    output = []
    count = 0
    for r, d, f in os.walk(directory):
        for file in f:
            one_percent = len(f) / 50
            if int(count % one_percent) == 0:
                os.system('cls')
                print('Reading: ' + directory)
                print(int((count + 1) / len(f) * 100), '%\t', '█' * int((count + 1) / one_percent))
            count += 1
            image = misc.imread(directory + '\\' + file, flatten = True)
            reduced_image = misc.imresize(image, (400, 400)).flatten()
            output.append(reduced_image)
    return output

root = "D:\\Desktop\\chest_xray"
dirs = ['\\train\\NORMAL',
        '\\train\\PNEUMONIA',
        '\\test\\NORMAL',
        '\\test\\PNEUMONIA',
        '\\val\\NORMAL',
        '\\val\\PNEUMONIA']

def img_input(dir1, dir2):
    X = input_images(root + dir1) + input_images(root + dir2)
    y = []
    for x in X:
        if 'bacteria' in x or 'virus' in x:
            y.append(1)
        else:
            y.append(0)
    return X, y

X_train, y_train = img_input(dirs[0], dirs[1])
X_test, y_test = img_input(dirs[2], dirs[3])
X_val, y_val = img_input(dirs[4], dirs[5])

params = {'activation' : ['logistic', 'tanh', 'relu'],
          'alpha' : [0.01, 0.001, 0.0001, 0.00001],
          'learning_rate' : ['constant', 'adaptive'],
          'learning_rate_init' : [0.01, 0.001, 0.0001]}
clf = MLPClassifier(warm_start = True, early_stopping = True)
grid = GridSearchCV(clf, params, cv = 5)
print("fitting")
grid.fit(X_train, y_train)
print(grid.best_estimator_)
grid.best_estimator_.fit(X_train, y_train)
print(grid.best_estimator.score(X_train, y_train))

#[print(len(k)) for k in train_normal_X]
#clf.fit([train_normal_X], train_normal_y)
#clf.score(test_normal_X, test_normal_y)

