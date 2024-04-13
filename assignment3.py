

# import required libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectFromModel

# load data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


# devide training data into features and target
X_train = train_data.drop('Education', axis=1)
y_train = train_data['Education']

# encode non-numerical labels into numerical ones.
label_encoder = LabelEncoder()
X_train_encoded = X_train.apply(label_encoder.fit_transform)
# split training data into 2 parts train and test
X_train, X_test, y_train, y_test = train_test_split(X_train_encoded, y_train, test_size=0.2, random_state=60)


# initializing randomforestclassifier
clf = RandomForestClassifier(random_state=60)
clf.fit(X_train, y_train)

# removing not important features from X-train.
feature_importances = clf.feature_importances_
sfm = SelectFromModel(clf, threshold=0.1)
X_train = sfm.fit_transform(X_train, y_train)


# make prediction using trained classifier
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
# calculate classification report including F1 score
report = classification_report(y_test, y_pred, zero_division=1)
print(report)

# load actual test data
test_data = pd.read_csv('test.csv')

# encode actual test data
test_data = test_data.apply(label_encoder.fit_transform)

# make predictions for actual test data
predictions = clf.predict(test_data)

# create a data frame comprising of ID and Education prediction
submission_df = pd.DataFrame({'ID': test_data['ID'], 'Education': predictions})

# put the data to file named submission.csv
submission_df.to_csv('submission.csv', index=False)

