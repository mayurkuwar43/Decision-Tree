
#Importing the dataset
        df = pd.read_csv("Downloads/churn.csv")
        df
# Checking the shape of the data
        df.shape
        df.dtypes
        df.columns
        df.head()
        df_desc = df.describe()
        df_desc
# Checking the variable type count in df
        df.dtypes.value_counts()
# Getting the uniQue Values
        for val in df:
            print(val, " ", df[val].unique().shape)

# dealing with UniQue, or same value columns.
        df.drop("customerID", axis=1, inplace=True)
        df.shape

# Converting the senior citizen column.
        lst =[]
        for val in df.SeniorCitizen:
            x = "Yes" if(val > 0.5) else "NO"
            lst.append(x)
        df["SeniorCitizen"] = lst

# Checking the single value value domination
        quasi_constant_feat = []
        for feature in df.columns:
            dominant = (df[feature].value_counts() / np.float(len(df))).sort_values(ascending=False).values[0]
            if dominant > 0.90:
                quasi_constant_feat.append(feature)

        print(quasi_constant_feat)


# Null Value analysis & treatment.
        df.isnull().any()
        # df.isnull()
        df.isnull().sum()

# Bar plot for tg variable
        sns.catplot(x="Churn", kind="count", data=df)

#Separating the target varaible
        Y = df.iloc[:,-1]
        Y.shape
        df.drop("Churn", axis=1, inplace=True)
        df.shape
# Get numrical and categorial column separate
        df_num = df.select_dtypes(include=['int64', 'float64'])
        df_factor = df.select_dtypes(include=['object'])

# -------------
# Follow the EDA rule of observing by plotting etc.
# ----------------------

# Rescaling Numerical Data
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        for val in df:
            if(df[val].dtypes in ['int64', 'float64']):
                df[[val]] = scaler.fit_transform(df[[val]])



# Convert the Categorical Data into dummy variabless'''

        df = pd.get_dummies(df, drop_first=False)
        # df.columns
        # df.shape
        # df.dtypes

# Converting categorical variable into factor.
        lst = df_num.columns
        for val in df:
            if(val not in lst):
                df[val] = df[val].astype("object")


        df.shape
        df.columns

# Training test split
        from sklearn.model_selection import train_test_split
        x_train,x_test,y_train,y_test = train_test_split(df, Y, random_state = 42,test_size = 0.3)

# Applying Decision tree classifier
        from sklearn.tree import DecisionTreeClassifier
        regressor = DecisionTreeClassifier(random_state=0)
        regressor.fit(x_train, y_train)

# Predicting values form decision tree
        y_pred = regressor.predict(x_test)

#Import scikit-learn metrics module for accuracy calculation
        from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)
        print(metrics.confusion_matrix(y_test, y_pred))

# save confusion matrix and slice into four pieces---- deep diving into confusion matrix
# Converting the categorical output into numerical output
        lst_test =[]
        for val in y_test:
            x = 1 if(val == "Yes") else 0
            lst_test.append(x)

        lst_pred =[]
        for val in y_pred:
            x = 1 if(val == "Yes") else 0
            lst_pred.append(x)

        confusion = metrics.confusion_matrix(y_test, y_pred)
        print(confusion)
        #[row, column]
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]

# Accuracy by calculation & built-in fuction
        print((TP + TN) / float(TP + TN + FP + FN))
        print(metrics.accuracy_score(lst_test, lst_pred))

# Classification Error
        classification_error = (FP + FN) / float(TP + TN + FP + FN) #Error
        print(classification_error*100)
        print((1 - metrics.accuracy_score(lst_test, lst_pred))*100)

# Sensitivity or recall score or tpr
        sensitivity = TP / float(FN + TP)
        print(sensitivity)
        print(metrics.recall_score(lst_test, lst_pred))

#Specificity
        specificity = TN / (TN + FP)
        print(specificity)

# False positive rate - FPR
        false_positive_rate = FP / float(TN + FP)
        print(false_positive_rate)
        print(1 - specificity)

# Precision
        precision = TP / float(TP + FP)
        print(precision)
        print(metrics.precision_score(lst_test, lst_pred))

"""Receiver Operating Characteristic (ROC)"""
# IMPORTANT: first argument is true values, second argument is predicted values
# roc_curve returns 3 objects fpr, tpr, thresholds
# fpr: false positive rate
# tpr: true positive rate
        fpr, tpr, thresholds = metrics.roc_curve(lst_test, lst_pred)
        plt.plot(fpr, tpr)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.rcParams['font.size'] = 12
        plt.title('ROC curve for diabetes classifier')
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.grid(True)


"""AUC - Area under Curve"""
# AUC is the percentage of the ROC plot that is underneath the curve:
# IMPORTANT: first argument is true values, second argument is predicted probabilities
        print(metrics.roc_auc_score(lst_test, lst_pred))


# F1 Score FORMULA - it is good if it closer to 1, the more the better.
        F1 = 2 * (precision * sensitivity) / (precision + sensitivity)



# Tree Visualization from graphviz --- needed graphviz msi.
        from IPython.display import Image
        from sklearn.tree import export_graphviz
        import pydotplus

# Export as dot file
        export_graphviz(regressor,
                        out_file='tree.dot',
                        feature_names = x_train.columns,
                        class_names = Y.unique(),
                        rounded = True, proportion = False,
                        precision = 2, filled = True)


# Convert to png using system command (requires Graphviz)
        from subprocess import call
        call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
        from IPython.display import Image
        Image(filename = 'tree.png')


# Display in python
        import matplotlib.pyplot as plt
        plt.figure(figsize = (14, 18))
        plt.imshow(plt.imread('tree.png'))
        plt.axis('off');
        plt.show();


# Another way of visulaization from tree package
        from IPython.display import Image
        from sklearn import tree
        import pydotplus


# Create DOT dta
        dot_data = tree.export_graphviz(regressor, out_file=None,
                                        feature_names=x_train.columns,
                                        class_names=Y.unique())

# Draw graph
        graph = pydotplus.graph_from_dot_data(dot_data)

#https://graphviz.gitlab.io/_pages/Download/Download_windows.html
# Show graph
        Image(graph.create_png())

# Create PDF
        graph.write_pdf("tree_graph.pdf")

# Create PNG
        graph.write_png("tree_graph.png")

