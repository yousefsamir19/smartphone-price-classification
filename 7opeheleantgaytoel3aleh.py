import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder , OrdinalEncoder
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
def clen_falsare3_mnh (df):
    df.columns = df.columns.str.strip().str.replace(" ",'_')
    df["Processor_Series"] = df["Processor_Series"].str.replace(" Gen1", ".1").str.replace(" Gen2", ".2").str.replace("Unknown", "35").astype(float)
    df["memory_card_size_GB"] = df["memory_card_size"].astype(str).str.replace("GB", "").str.replace("TB", "*1000").map(lambda X: eval(X))
    df["memory_card_size_GB"] = df["memory_card_size_GB"].astype(int)
    df.drop(columns="memory_card_size", inplace=True)
    df["os_version"] = df["os_version"].str.replace("v", "").str.replace(".", "", 1).astype(float)
    df["os_version"] = df["os_version"].apply(lambda x: x / 10 if x > 17 else x)

    outliers_col = ['rating', 'Processor_Series', 'Core_Count', 'Clock_Speed_GHz',
                    'RAM_Size_GB', 'Storage_Size_GB', 'battery_capacity', 'fast_charging_power',
                    'Screen_Size', 'Resolution_Width', 'Resolution_Height', 'Refresh_Rate',
                    'primary_rear_camera_mp', 'primary_front_camera_mp', 'num_front_cameras', 'memory_card_size_GB']

    outliers_col_clean = [col.strip() for col in outliers_col]
    for c in outliers_col_clean:
        df[f"log_{c}"] = np.log1p(df[c])

    df['performance_score'] = df['log_Core_Count'] * df['log_Clock_Speed_GHz'] * (df['log_RAM_Size_GB'] / 4)
    df['camera_quality_score'] = (df['log_primary_rear_camera_mp'] * 0.7 + df['log_primary_front_camera_mp'] * 0.3)

    df.drop(columns=outliers_col_clean, inplace=True)

    replace_dict = {"Poco": "POCO", "Oppo": "OPPO", "itel": "Itel", "Motorola Edge": "Motorola", }
    df['brand'] = df['brand'].replace(replace_dict)

    tiers_order = [
        ['Unknown', 'Budget', 'Low-End', 'Mid-Range', 'High-End', 'Flagship'],
        ['Unknown', 'Budget', 'Low-End', 'Mid-Range', 'High-End', 'Flagship']
    ]
    oe = OrdinalEncoder(categories=tiers_order)
    df[['Performance_Tier', 'RAM_Tier']] = oe.fit_transform(df[['Performance_Tier', 'RAM_Tier']])



    return df





df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_test = clen_falsare3_mnh(df_test)
df_train = clen_falsare3_mnh(df_train)
print("Train Shape:", df_train.shape)
print("Test Shape:", df_test.shape)

df_train.drop_duplicates(inplace=True)

df_train["Performance_Tier"].value_counts()
df_train["RAM_Tier"].value_counts()
modeRam = df_train["RAM_Tier"].mode()[0]
df_train["RAM_Tier"] = df_train["RAM_Tier"].replace("Unknown", modeRam)
df_test["RAM_Tier"] = df_test["RAM_Tier"].replace("Unknown", modeRam)


modeRam = df_train["Notch_Type"].mode()[0]
df_train["Notch_Type"] = df_train["Notch_Type"].str.replace("Unknown", modeRam)
df_test["Notch_Type"] = df_test["Notch_Type"].str.replace("Unknown", modeRam)

binary_columns=['price','Dual_Sim', '4G', '5G', 'Vo5G', 'NFC', 'IR_Blaster','memory_card_support']
le=LabelEncoder()
for col in binary_columns:
    df_train[col] = le.fit_transform(df_train[col])
    df_test[col] = le.transform(df_test[col])



plt.figure(figsize=(15,10))
numeric_corr = df_train.select_dtypes(include='number').corr()
sns.heatmap(numeric_corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap (numeric features)')
plt.show()

min_frquency = 10

brand_counts = df_train['brand'].value_counts()
brands_to_replace = brand_counts[brand_counts < min_frquency].index
df_train['brand'] = df_train['brand'].replace(brands_to_replace, 'Other')
df_test['brand'] = df_test['brand'].replace(brands_to_replace, 'Other')

count= df_train['Processor_Brand'].value_counts()
processor_brand_replace = count[count < min_frquency].index
df_train['Processor_Brand'] = df_train['Processor_Brand'].replace(processor_brand_replace, 'Other')
df_test['Processor_Brand'] = df_test['Processor_Brand'].replace(processor_brand_replace, 'Other')


count_type=df_train['Notch_Type'].value_counts()
notch_type_replace = count_type[count_type < min_frquency].index
df_train['Notch_Type'] = df_train['Notch_Type'].replace(notch_type_replace, 'Other')
df_test['Notch_Type'] = df_test['Notch_Type'].replace(notch_type_replace, 'Other')

count_os=df_train['os_name'].value_counts()
os_replace = count_os[count_os < min_frquency].index
df_train['os_name'] = df_train['os_name'].replace(os_replace, 'Other')
df_test['os_name'] = df_test['os_name'].replace(os_replace, 'Other')



colms=['Processor_Brand','Performance_Tier','RAM_Tier','Notch_Type','os_name','brand']
for c in colms:
    print(f"{c}: {df_train[c].unique()}")




one_hot_cols = ['Processor_Brand', 'Notch_Type', 'os_name']
ohn = OneHotEncoder(drop='first', sparse_output=False)
encoded_train = ohn.fit_transform(df_train[one_hot_cols])
encoded_train_cols = ohn.get_feature_names_out(one_hot_cols)
encoded_train_df = pd.DataFrame(encoded_train, columns=encoded_train_cols)



encoded_test = ohn.transform(df_test[one_hot_cols])
encoded_test_cols = ohn.get_feature_names_out(one_hot_cols)
encoded_test_df = pd.DataFrame(encoded_test, columns=encoded_test_cols)


df_train.reset_index(drop=True, inplace=True)
df_train = pd.concat([df_train, encoded_train_df], axis=1)
df_train.drop(columns=one_hot_cols, inplace=True)

df_test.reset_index(drop=True, inplace=True)
df_test = pd.concat([df_test, encoded_test_df], axis=1)
df_test.drop(columns=one_hot_cols, inplace=True)



te = TargetEncoder(cols=['brand'], min_samples_leaf=20, smoothing=10)
df_train['brand'] = te.fit_transform(df_train['brand'], df_train['price'])
df_test['brand'] = te.transform(df_test['brand'])


plt.figure(figsize=(35,30))


numeric_corr = df_train.select_dtypes(include='number').corr()
sns.heatmap(numeric_corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap (numeric features)')
plt.show()


X_train = df_train.drop('price', axis=1)
y_train = df_train['price']


X_test = df_test.drop('price', axis=1)
y_test = df_test['price']

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

mi = mutual_info_regression(X_train, y_train)
mi_series = pd.Series(mi, index=X_train.columns)
mi_series = mi_series.sort_values(ascending=False)

print(mi_series.head(20))

plt.figure(figsize=(10,8))
mi_series.head(20).plot(kind='barh')
plt.show()



#modeling



X_train = df_train.drop('price', axis=1)
y_train = df_train['price']

X_test = df_test.drop('price', axis=1)
y_test = df_test['price']


top_features = mi_series.head(30).index.tolist()

X_train = X_train[top_features]
X_test = X_test[top_features]
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



models = {
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(),
    "svm": SVC(probability=True),
    "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "knn": KNeighborsClassifier()
}

parameters = {
    "decision_tree": {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': [None, 'balanced']
    },
    "random_forest": {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False],
        'class_weight': [None, 'balanced']
    },
    "svm": {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf', 'poly'],
        'degree': [2, 3],
        'class_weight': [None, 'balanced']
    },
    "xgboost": {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2],
        'scale_pos_weight': [1]
    },
    "knn": {
        'n_neighbors': [3, 5, 7, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'p': [1, 2]
    }
}


def run_grid_search(models, parameters, X_train, y_train, X_test, y_test):
    results = {}
    for model_name, model in models.items():
        print(f"\nRunning GridSearchCV for {model_name}...")

        grid_search = GridSearchCV(model, parameters[model_name], cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

        grid_search.fit(X_train, y_train)
        results[model_name] = grid_search

        print(f"Best parameters for {model_name}: {grid_search.best_params_}")


        y_pred = grid_search.predict(X_test)

        print(f"--- Results for {model_name} ---")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("=" * 40)


run_grid_search(models, parameters, X_train, y_train, X_test, y_test)


DT=DecisionTreeClassifier(class_weight='balanced', criterion='gini', max_depth=20, max_features=None, min_samples_leaf=2, min_samples_split=2)
DT.fit(X_train,y_train)
y_pred=DT.predict(X_test)
print("classification report")
print(classification_report(y_test,y_pred))
#plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
# import joblib
# joblib.dump(DT, 'smartphone_price_classifier_model.pkl')



#SVM=SVC(C=1, class_weight='balanced', degree=3, gamma='scale', kernel='poly')   with 20 features
SVM=SVC(C=10, class_weight=None, degree=2, gamma='scale', kernel='rbf')   #with 30 features

SVM.fit(X_train,y_train)
y_pred=SVM.predict(X_test)
print("classification report")
print(classification_report(y_test,y_pred))
#plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
# plt.show()
# joblib.dump(SVM, 'smartphone_price_svm_classifier_model.pkl')

# randooooooom forest

rf =RandomForestClassifier(bootstrap=True, class_weight='balanced', max_depth=20, max_features='sqrt', min_samples_leaf=1, min_samples_split=2, n_estimators=100)
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
print("classification report")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('predicted')
plt.ylabel('actual')
plt.title('confusion matrix')
plt.show()




#xgboost


XGB = XGBClassifier(colsample_bytree=0.8, gamma=0, learning_rate=0.1, max_depth=3, n_estimators=200, scale_pos_weight=1, subsample=0.7, use_label_encoder=False, eval_metric='logloss')

XGB.fit(X_train, y_train)
y_pred = XGB.predict(X_test)

print("classification report")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - XGBoost')
plt.show()