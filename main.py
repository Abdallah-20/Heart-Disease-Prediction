from sklearn.feature_selection import SelectPercentile, chi2,GenericUnivariateSelect,SelectKBest, f_classif
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics,svm
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import warnings
df = pd.read_csv("heart.csv")
print(df.shape)
# cleaning:
    # check and remove duplicated rows:
if df.duplicated:
    df.drop_duplicates(inplace=True)
    # check and remove nulls:
if df.isnull:
    df.dropna(inplace=True)
def Naive_Bayes(dataframe):
    x = df.iloc[:, 0:13]
    print(x)
    y = df.iloc[:, 13]
    print(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    res = gnb.predict(x_test)
    print(res)

    accuracy = accuracy_score(y_test, res)
    print("Naive_Bayes Accuracy is: " + str(round(accuracy*100, 1)) + " %")


def DecisionTree(dataframe):


    x = dataframe.drop("sex", axis=1)
    y = dataframe["sex"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=1000)
    x_train_dt = x_train.drop("fbs", axis=1)
    x_train_dt = x_train_dt.drop("trestbps", axis=1)
    x_train_dt = x_train_dt.drop("chol", axis=1)
    x_train_dt = x_train_dt.drop("age", axis=1)
    x_train_dt = x_train_dt.drop("oldpeak", axis=1)

    x_test_dt = x_test.drop("fbs", axis=1)
    x_test_dt = x_test_dt.drop("trestbps", axis=1)
    x_test_dt = x_test_dt.drop("chol", axis=1)
    x_test_dt = x_test_dt.drop("age", axis=1)
    x_test_dt = x_test_dt.drop("oldpeak", axis=1)

    sn.barplot(data=dataframe, x="sex", y="chol", hue="target", palette='spring')
    plt.show()

    DTREE = DecisionTreeClassifier(criterion='entropy', random_state=0)
    DTREE.fit(x_train_dt, y_train)
    y_pred_dt1 = DTREE.predict(x_test_dt)
    print("Accuracy of decision Tree after removing features:: ", str(round(metrics.accuracy_score(y_test, y_pred_dt1)*100))+"%")
def Logistic(dataframe):
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    # -----------1---------------
    array = dataframe.values
    X = array[:, 0:13]
    Y = array[:, 13]
    # -----------2---------------
    X = dataframe.drop("target", axis=1)
    Y = dataframe.target
    # -----------3---------------
    X = dataframe.iloc[:, 0:13]
    Y = dataframe.iloc[:, 13]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
    # univariate_method------feature_selection:
    best_feature = SelectKBest(f_classif, k=10)
    x_train_kbest = best_feature.fit_transform(X_train, y_train)
    select_features = pd.DataFrame(best_feature.inverse_transform(x_train_kbest), index=X_train.index,
                                   columns=X_train.columns)
    select_col = select_features.columns[select_features.var() != 0]
    print('Selected Features:', select_col)

    # -----------------------------Logistic Regression----------------------
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)

    # ----------------------------plot-------------------------------
    x = dataframe.trestbps
    plt.scatter(x, Y)
    plt.title('Scatter Plot of Logistic Regression')
    plt.show()

    print("-----------------------------------------------------------")
    Accuracy_score = round(accuracy_score(y_pred, y_test) * 100, 1)
    print("The accuracy score: " + str(Accuracy_score) + " %")
    print("-----------------------------------------------------------")

def RandomForest(dataframe):
    # split data into train and test:
    y = dataframe['target']
    x = dataframe.drop('target', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    # preprocessing:
    sc_x = StandardScaler()
    tr_tree = sc_x.fit_transform(x_train)
    te_tree = sc_x.transform(x_test)
    # RandomForest:
    classifier = RandomForestClassifier(n_estimators=600)
    classifier2 = RandomForestClassifier(n_estimators=600)
    classifier.fit(x_train, y_train)
    classifier2.fit(tr_tree, y_train)
    # accuracy:
    rf_pred = classifier.predict(x_test)
    predict = classifier2.predict(te_tree)
    RF = metrics.accuracy_score(y_test, predict)
    RandomForest_accuracy = metrics.accuracy_score(y_test, rf_pred)
    print("Random Forest Accuracy: ", str(round(RandomForest_accuracy*100))+"%")
    print("Random Forest Accuracy after: ", str(round(RF*100))+"%")
    # Feature Selection:
    features = GenericUnivariateSelect(score_func=chi2, mode='k_best', param=3)
    new_features = features.fit_transform(x, y)
    print("parameters that play a vital role in determining the health condition of people’s hearts :",features.get_feature_names_out())
    # visualization:
    confusion_matrix = pd.crosstab(y_test, rf_pred, rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix, annot=True)

    plt.show()
def SVM(dataframe):
    # split data into train and test:
    train_columns = dataframe.iloc[:, 0:13]
    test_column = dataframe['target']
    X_train, X_test, Y_train, Y_test = train_test_split(train_columns, test_column, test_size=0.20, random_state=0)
    # preprocessing:
    sc_x = StandardScaler()
    train = sc_x.fit_transform(X_train)
    test = sc_x.transform(X_test)
    # SVM:
    data = svm.SVC(kernel='linear')
    d2 = svm.SVC(kernel='linear')
    data.fit(X_train, Y_train)
    d2.fit(train, Y_train)
    # accuracy:
    svm_predict = data.predict(X_test)
    predicto = d2.predict(test)
    svm_accuracy = metrics.accuracy_score(Y_test, svm_predict)
    svm_sc = metrics.accuracy_score(Y_test, predicto)
    print("SVM Accuracy: ",  str(round(svm_accuracy*100))+"%")
    print("SVM Accuracy after preprocessing: ", str(round(svm_sc*100))+"%")
    # Feature Selection:
    affective_parameters = SelectPercentile(score_func=chi2, percentile=30)
    x = affective_parameters.fit_transform(train_columns, test_column)
    print("parameters that play a vital role in determining the health condition of people’s hearts :",
          affective_parameters.get_feature_names_out())
    # visualization:
    confusion_matrix = pd.crosstab(Y_test, svm_predict, rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix, annot=True)

    plt.show()

def KNN(dataframe):
    x = dataframe.drop(['target'], axis=1)  # ....spliting data
    y = dataframe['target']
    scaler = StandardScaler()  # ...data normlization
    normalised_features = scaler.fit_transform(x)
    X_train, X_test, y_train, y_test = train_test_split(normalised_features, y, test_size=0.2, random_state=0)
    KNN_model = KNeighborsClassifier()
    KNN_model.fit(X_train, y_train)
    y_pred = KNN_model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("{} KNN Score: {:.2f}%".format('THE', KNN_model.score(X_test, y_test) * 100))
    scores = []
    for n in list(range(1, 20)):
        KNN_model2 = KNeighborsClassifier(n_neighbors=n)
        KNN_model2.fit(X_train, y_train)
        scores.append(KNN_model2.score(X_test, y_test))
    plt.plot(list(range(1, 20)), scores)
    plt.xlabel('Number of neighbours')
    plt.ylabel('Knn score')
    plt.title('The change in algorithm score as the number of neighbours was changed')
    plt.show()
    accuracy = max(scores) * 100
    print("Best KNN Score is: {:.2f}%".format(accuracy))
def correlation(dataframe):
    print(
        "1.Spearman correlation"'\n'"2.Pearson"'\n'"3.Histogram")
    response = int(input("Please Choose Your Option \n"))
    if response == 1:
        #*********************spearman****************#
        SpearmanCorrleation = dataframe.corr(method='spearman')
        fig = plt.subplots(figsize=(12, 6))
        sn.heatmap(SpearmanCorrleation, vmin=0, vmax=1, cmap="Blues", data=dataframe.corr(), annot=True)
        plt.title("Spearman Correlation")
        sn.catplot(data=dataframe, x='sex', y='age', hue='target', palette='husl')
        plt.show()
        cor_target = SpearmanCorrleation["target"]
        # Selecting highly correlated features
        k = cor_target[cor_target > 0]
        UsedFeatures = k[k != 1]
        print("Used Features", '\n', UsedFeatures)
    elif response == 2:
        # *********************Pearson****************#
        PearsonCorrleation = dataframe.corr(method='pearson')
        fig = plt.subplots(figsize=(12, 6))
        sn.heatmap(PearsonCorrleation, vmin=0, vmax=1, data=dataframe.corr(), annot=True)
        plt.title("Pearson Correlation")
        plt.show()
        # Correlation with output variable
        cor_target = PearsonCorrleation["target"]
        # Selecting highly correlated features
        k = cor_target[cor_target > 0]
        UsedFeatures = k[k != 1]
        print("Used Features", '\n', UsedFeatures)
    elif response == 3:
        df.hist(figsize=(12, 12), layout=(5, 3))
    else:
        print("Wrong Choice")
choice =1
while(choice==1):
    print('\t''\t''\t''\t''\t''\t''\t''\t''\t''\t''\t''\t''\t''\t''\t''\t''\t'"____________Menu___________")
    print("1.DecisionTree"'\n'"2.Naive_Bayes"'\n'"3.KNN"'\n'"4.RandomForest"'\n'"5.SVM"'\n'"6.Logistic"'\n'"7.Correlation matrices")
    response = int(input("Please Choose Your Option \n"))
    if response==1:
        DecisionTree(df)
    elif response==2:
        Naive_Bayes(df)
    elif response==3:
        KNN(df)
    elif response==4:
        RandomForest(df)
    elif response==5:
        SVM(df)
    elif response==6:
        Logistic(df)
    elif response == 7:
        correlation(df)
    else:
        print("Wrong Choice")

