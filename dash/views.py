import os
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.offline as opy
from collections import defaultdict
from django.views.decorators.clickjacking import xframe_options_exempt
from pathlib import Path
from pandas_profiling import ProfileReport
from pandas_profiling.utils.cache import cache_file
from scipy.sparse import csr_matrix, issparse
from django.contrib import messages
from django.shortcuts import render, redirect
from django.views.generic import TemplateView, CreateView
from django.urls import reverse_lazy
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib.auth.decorators import login_required
from django.views.generic.base import ContextMixin
from missingpy import MissForest
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,\
    StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler,\
    PowerTransformer, QuantileTransformer, minmax_scale, Normalizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from dash.models import CurrentFile, Prepross, CSV
from .forms import UserUpdateForm, ProfileUpdateForm, CSVForm


# ------------------ UPLOADING CSV FILE ------------------------- #
class UploadBookView(CreateView):
    model = CSV
    form_class = CSVForm
    success_url = reverse_lazy('index')
    template_name = 'dash/upload.html'


# ------------------ SELECTING CSV FILE ------------------------- #
class index(TemplateView):
    template_name = 'dash/index.html'

    def post(self, request, **kwargs):
        context = {}
        if request.method == 'POST':
            current_file = CurrentFile(filename=list(request.POST.keys())[1])
            current_file.save()
            context['fileselected'] = list(request.POST.keys())[1]
            context['test'] = 'a'
            context['title'] = 'Select File'
            return render(request, 'dash/index.html', context)


# ------------------ Explore CSV FILE ------------------------- #
@login_required
def explore(request):
    try:
        file_name = CurrentFile.objects.order_by('-id')[0].filename
        df = pd.read_csv(os.path.join('Media\csvfiles', file_name), low_memory=False, delimiter=',', nrows=2000,
                         encoding="ISO-8859-1")
        profile = ProfileReport(
            df,
            title="Dataset Information",
            correlations={"cramers": {"calculate": False}},
        )
        profile.to_file(output_file=Path("dash/templates/dash/output.html"))
    except:
        print('sdfsdf')
    return render(request, 'dash/explore.html')


@xframe_options_exempt
def explore_data(request):
    return render(request, 'dash/output.html', {'title': 'explore'})


# ------------------ PREPROCESSING  ------------------------- #
class prepross(TemplateView, TransformerMixin, ContextMixin):
    template_name = 'dash/preprocessing.html'

    # model = Prepross
    # fields = ('file_name','coltype','Xvars','Yvar','onehot','featscaling','Drop_NA')
    def __init__(self, **kwargs):
        """Impute missing values.
        Columns of dtype object are imputed with the most frequent value
        in column.
        Columns of other types are imputed with mean of column.
        """
        super().__init__(**kwargs)

    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
                               if X[c].dtype == np.dtype('O') else X[c].median() for c in X], index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        try:
            file_name = CurrentFile.objects.order_by('-id')[0].filename
            df1 = pd.read_csv(os.path.join('Media\csvfiles', file_name), low_memory=False, delimiter=',', nrows=2000, encoding="ISO-8859-1")
            # Infer Data types
            df = df1.infer_objects()
            # profile = ProfileReport(df, title='DATA INFORMATION', minimal=True, html={'style': {'full_width': True}})
            # profile.to_file(output_file="dash/output.html")
            count_nan = len(df) - df.count()
            row_count = df.count()[0]
            file_type = pd.concat([df.dtypes, count_nan, df.nunique()], axis=1)
            file_type.columns = ("Type", "NA count", "Count distinct")
            context['DataFrame'] = df
            context['file_type'] = file_type
            context['title'] = 'Preprocess'
            context['row_count'] = row_count
        except:
            file_name = 'Please select one file'
        context['file_name'] = file_name
        return context

    def post(self, request, **kwargs):
        global df_X1, df_X2, df_X3, df_Y1, df_Y2, df_Y3, ALL_Y_DF, ALL_X_DF, data_numeric_x, data_numeric_y
        df_X1 = pd.DataFrame()
        df_X2 = pd.DataFrame()
        df_X3 = pd.DataFrame()
        df_Y1 = pd.DataFrame()
        df_Y2 = pd.DataFrame()
        df_Y3 = pd.DataFrame()
        data_numeric_x = pd.DataFrame()
        data_numeric_y = pd.DataFrame()
        context = self.get_context_data(**kwargs)
        if request.method == 'POST':
            # Get dataframe and change data type
            file_name = CurrentFile.objects.order_by('-id')[0].filename
            coltype = request.POST.getlist('coltype')
            coltype = dict([i.split(':', 1) for i in coltype])
            # Reading Dataframe
            df = pd.read_csv(os.path.join('Media\csvfiles', file_name), low_memory=False, delimiter=',', nrows=2000, dtype=coltype, encoding="ISO-8859-1")

            categorical_columns = []
            numeric_columns = []
            for c in df.columns:
                if df[c].map(type).eq(str).any():  # check if there are any strings in column
                    categorical_columns.append(c)
                else:
                    numeric_columns.append(c)
            # create two DataFrames, one for each data type
            data_numeric = df[numeric_columns]  # Numerical List.
            data_categorical = pd.DataFrame(df[categorical_columns])  # Categorical List.
            # Imputation of Categorical Values by Mean
            data_categorical = pd.DataFrame(prepross().fit_transform(data_categorical), columns=data_categorical.columns)
            df = pd.concat([data_numeric, data_categorical], axis=1)
            # row_count = df.count()[0]

            # request from user form which type of encoding need for each column
            encoding = request.POST.getlist('encoding')
            col_encode = [aa for aa in encoding if ":column_encoder" in aa]
            col_encode_names_list = [i.split(':', 1)[0] for i in col_encode]  # List of Column Transformer.
            col_label = [bb for bb in encoding if ":label_encoder" in bb]
            col_label_names_list = [i.split(':', 1)[0] for i in col_label]   # List of Label Encoder.
            col_no_label = [cc for cc in encoding if ":no_encoding" in cc]
            col_no_label_names_list = [i.split(':', 1)[0] for i in col_no_label]  # List of No Encoding.
            col_all = col_encode_names_list + col_label_names_list + col_no_label_names_list  # Adding all List Together
            df = df[col_all]  # Passing to Dataframe.
            col_encode_names = ', '.join(col_encode_names_list)
            col_label_names = ', '.join(col_label_names_list)
            col_no_label_names = ', '.join(col_no_label_names_list)

            # Selecting  X and Y and Column Remove
            assvar = request.POST.getlist('assvar')
            xcols0 = [s for s in assvar if ":X" in s]
            xcols = [i.split(':', 1)[0] for i in xcols0]  # Select X cols from user form
            ycol0 = [s for s in assvar if ":y" in s]
            ycol = [i.split(':', 1)[0] for i in ycol0]  # Select y cols from user form
            x_cols = df[xcols]
            y_col = df[ycol]
            cols = xcols + ycol  # combining Both X and Y
            df = df[cols]  # Creating a DataFrame from combined Cols
            xcols = ', '.join(xcols)
            ycol = ', '.join(ycol)
            duplicate_rows = request.POST.getlist('duplicate_rows')
            duplicate_rows = ', '.join(duplicate_rows)
            missing = request.POST.getlist('missingvalues')
            missing = ', '.join(missing)
            imputing = request.POST.getlist('imputing')
            imputing = ', '.join(imputing)
            trainingset_s = request.POST.getlist('trainingset_size')
            trainingset_s = ', '.join(trainingset_s)
            testset_s = 100 - int(trainingset_s)
            scaling = request.POST.getlist('scaling')
            scaling = ', '.join(scaling)
            norms = request.POST.getlist('norms')
            norms = ', '.join(norms)
            models = request.POST.getlist('models')
            models = ', '.join(models)
            # Taking care of missing data
            if duplicate_rows == "Keep_dup":
                context['dup_msg'] = 'DataFrame Having Duplicate Rows'
            if duplicate_rows == "Drop_dup":
                df = df.drop_duplicates()
            if missing == "Keep_NAN":
                if len(df) != len(df.dropna()):
                    context['selecty'] = 'Null Values are imputed'
            if missing == "Drop_NAN":
                df = df.dropna()
            if imputing == "mean" or\
                    imputing == "median" or\
                    imputing == "most_frequent" or\
                    imputing == "constant":
                impute = imputing
                categorical_columns_x = []
                numeric_columns_x = []
                for c in x_cols.columns:
                    if x_cols[c].map(type).eq(str).any():  # check if there are any strings in column
                        categorical_columns_x.append(c)
                    else:
                        numeric_columns_x.append(c)
                # create two DataFrames, one for each data type
                data_numeric_x = df[numeric_columns_x]
                data_categorical_x = pd.DataFrame(df[categorical_columns_x])
                if len(data_numeric_x.columns) >= 1:
                    imp = SimpleImputer(missing_values=np.nan, strategy=impute)
                    # only apply imputes to numeric columns
                    data_numeric_x = pd.DataFrame(imp.fit_transform(data_numeric_x), columns=data_numeric_x.columns)
                    # you could do something like one-hot-encoding of data_categorical here
                    # join the two masked data frames back together
                    x_cols = pd.concat([data_numeric_x, data_categorical_x], axis=1)
                categorical_columns_y = []
                numeric_columns_y = []
                for c in y_col.columns:
                    if y_col[c].map(type).eq(str).any():  # check if there are any strings in column
                        categorical_columns_y.append(c)
                    else:
                        numeric_columns_y.append(c)
                # create two DataFrames, one for each data type
                data_numeric_y = df[numeric_columns_y]
                data_categorical_y = pd.DataFrame(df[categorical_columns_y])
                if len(data_numeric_y.columns) >= 1:
                    imp = SimpleImputer(missing_values=np.nan, strategy=impute)
                    # only apply imputes to numeric columns
                    data_numeric_y = pd.DataFrame(imp.fit_transform(data_numeric_y), columns=data_numeric_y.columns)
                    # you could do something like one-hot-encoding of data_categorical here
                    # join the two masked data frames back together
                    y_col = pd.concat([data_numeric_y, data_categorical_y], axis=1)

            if imputing == "KNN":
                categorical_columns_x = []
                numeric_columns_x = []
                for c in x_cols.columns:
                    if x_cols[c].map(type).eq(str).any():  # check if there are any strings in column
                        categorical_columns_x.append(c)
                    else:
                        numeric_columns_x.append(c)
                # create two DataFrames, one for each data type
                data_numeric_x = df[numeric_columns_x]
                data_categorical_x = pd.DataFrame(df[categorical_columns_x])
                if len(data_numeric_x.columns) >= 1:
                    imp = KNNImputer(n_neighbors=5, weights="uniform")
                    # only apply imputes to numeric columns
                    data_numeric_x = pd.DataFrame(imp.fit_transform(data_numeric_x), columns=data_numeric_x.columns)
                    # you could do something like one-hot-encoding of data_categorical here
                    # join the two masked data frames back together
                    x_cols = pd.concat([data_numeric_x, data_categorical_x], axis=1)
                categorical_columns_y = []
                numeric_columns_y = []
                for c in y_col.columns:
                    if y_col[c].map(type).eq(str).any():  # check if there are any strings in column
                        categorical_columns_y.append(c)
                    else:
                        numeric_columns_y.append(c)
                # create two DataFrames, one for each data type
                data_numeric_y = df[numeric_columns_y]
                data_categorical_y = pd.DataFrame(df[categorical_columns_y])
                if len(data_numeric_y.columns) >= 1:
                    imp = KNNImputer(n_neighbors=5, weights="uniform")
                    # only apply imputes to numeric columns
                    data_numeric_y = pd.DataFrame(imp.fit_transform(data_numeric_y), columns=data_numeric_y.columns)
                    # you could do something like one-hot-encoding of data_categorical here
                    # join the two masked data frames back together
                    y_col = pd.concat([data_numeric_y, data_categorical_y], axis=1)
            if imputing == "RDF":
                categorical_columns_x = []
                numeric_columns_x = []
                for c in x_cols.columns:
                    if x_cols[c].map(type).eq(str).any():  # check if there are any strings in column
                        categorical_columns_x.append(c)
                    else:
                        numeric_columns_x.append(c)
                # create two DataFrames, one for each data type
                data_numeric_x = df[numeric_columns_x]
                data_categorical_x = pd.DataFrame(df[categorical_columns_x])
                if len(data_numeric_x.columns) >= 1:
                    imp = MissForest(max_iter=10, decreasing=False, missing_values=np.nan,
                                     copy=True, n_estimators=100, criterion=('mse', 'gini'),
                                     max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                     min_weight_fraction_leaf=0.0, max_features='auto',
                                     max_leaf_nodes=None, min_impurity_decrease=0.0,
                                     bootstrap=True, oob_score=False, n_jobs=-1, random_state=None,
                                     verbose=0, warm_start=False, class_weight=None)
                    # only apply imputes to numeric columns
                    data_numeric_x = pd.DataFrame(imp.fit_transform(data_numeric_x), columns=data_numeric_x.columns)
                    # you could do something like one-hot-encoding of data_categorical here
                    # join the two masked data frames back together
                    x_cols = pd.concat([data_numeric_x, data_categorical_x], axis=1)
                categorical_columns_y = []
                numeric_columns_y = []
                for c in y_col.columns:
                    if y_col[c].map(type).eq(str).any():  # check if there are any strings in column
                        categorical_columns_y.append(c)
                    else:
                        numeric_columns_y.append(c)
                # create two DataFrames, one for each data type
                data_numeric_y = df[numeric_columns_y]
                data_categorical_y = pd.DataFrame(df[categorical_columns_y])
                print(data_numeric_y.columns)
                if len(data_numeric_y.columns) >= 1:
                    imp = MissForest(max_iter=10, decreasing=False, missing_values=np.nan,
                                     copy=True, n_estimators=100, criterion=('mse', 'gini'),
                                     max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                     min_weight_fraction_leaf=0.0, max_features='auto',
                                     max_leaf_nodes=None, min_impurity_decrease=0.0,
                                     bootstrap=True, oob_score=False, n_jobs=-1, random_state=None,
                                     verbose=0, warm_start=False, class_weight=None)
                    # only apply imputes to numeric columns
                    data_numeric_y = pd.DataFrame(imp.fit_transform(data_numeric_y), columns=data_numeric_y.columns)
                    # you could do something like one-hot-encoding of data_categorical here
                    # join the two masked data frames back together
                    y_col = pd.concat([data_numeric_y, data_categorical_y], axis=1)
            # Return error if columns are not selected
            if any([item in x_cols for item in col_encode_names_list]):
                A = list(x_cols)
                B = col_encode_names_list
                C = list(set(A) & set(B))
                col_encode_names_vals_x = pd.DataFrame(x_cols[C])
                ohe = OneHotEncoder(handle_unknown='ignore')
                ct = make_column_transformer((ohe, C), remainder='passthrough')
                ctt = ct.fit_transform(col_encode_names_vals_x)
                if issparse(ctt):
                    df_X1 = pd.DataFrame.sparse.from_spmatrix(ctt)
                else:
                    df_X1 = pd.DataFrame(ctt)

                # print("column transformer - X")
                # print(df_X1)

            if any([item in y_col for item in col_encode_names_list]):
                A = list(y_col)
                B = col_encode_names_list
                C = list(set(A) & set(B))
                col_encode_names_vals_y = pd.DataFrame(y_col[C])
                ohe = OneHotEncoder(handle_unknown='ignore')
                ct = make_column_transformer((ohe, C), remainder='passthrough')
                ctt = ct.fit_transform(col_encode_names_vals_y)
                if issparse(ctt):
                    df_Y1 = pd.DataFrame.sparse.from_spmatrix(ctt)
                else:
                    df_Y1 = pd.DataFrame(ctt)
                # print("column transformer - Y")
                # print(df_Y1)

            if any([item in x_cols for item in col_label_names_list]):
                E = list(x_cols)
                F = col_label_names_list
                G = list(set(E) & set(F))
                col_label_names_vals_x = pd.DataFrame(x_cols[G])
                d = defaultdict(LabelEncoder)
                df_X2 = col_label_names_vals_x.apply(lambda x: d[x.name].fit_transform(x))
                # print("Label transformer - X")
                # print(df_X2)

            if any([item in y_col for item in col_label_names_list]):
                E = list(y_col)
                F = col_label_names_list
                G = list(set(E) & set(F))
                col_label_names_vals_y = pd.DataFrame(y_col[G])
                d = defaultdict(LabelEncoder)
                df_Y2 = col_label_names_vals_y.apply(lambda x: d[x.name].fit_transform(x))
                # print("Label transformer - Y")
                # print(df_Y2)

            if any([item in x_cols for item in col_no_label_names_list]):
                H = list(x_cols)
                I = col_no_label_names_list
                J = list(set(H) & set(I))
                col_no_label_names_vals_x = pd.DataFrame(x_cols[J])
                df_X3 = col_no_label_names_vals_x
                # print("NOOO transformer - X")
                # print(df_X3)

            if any([item in y_col for item in col_no_label_names_list]):
                H = list(y_col)
                I = col_no_label_names_list
                J = list(set(H) & set(I))
                col_no_label_names_vals_y = pd.DataFrame(y_col[J])
                df_Y3 = col_no_label_names_vals_y
                # print("NOOO transformer - Y")
                # print(df_Y3)
            ALL_X_DF = pd.concat([df_X1, df_X2, df_X3], axis=1)
            ALL_Y_DF = pd.concat([df_Y1, df_Y2, df_Y3], axis=1)
            testset = (testset_s/100)
            X_train, X_test, y_train, y_test = train_test_split(ALL_X_DF, ALL_Y_DF, test_size=testset, random_state=15)
            if scaling == "SS_scale":
                ss_scaler = StandardScaler()
                X_train = ss_scaler.fit_transform(X_train)
                X_test = ss_scaler.fit_transform(X_test)
            if scaling == "MMS_scale":
                mms_scaler = MinMaxScaler()
                X_train = mms_scaler.fit_transform(X_train)
                X_test = mms_scaler.fit_transform(X_test)
            if scaling == "MAS_scale":
                mas_scaler = MaxAbsScaler()
                X_train = mas_scaler.fit_transform(X_train)
                X_test = mas_scaler.fit_transform(X_test)
            if scaling == "RS_scale":
                mas_scaler = RobustScaler(quantile_range=(0.1, 0.9))
                X_train = mas_scaler.fit_transform(X_train)
                X_test = mas_scaler.fit_transform(X_test)
            if scaling == "PTYJ_scale":
                ptyj_scaler = PowerTransformer(method='yeo-johnson')
                X_train = ptyj_scaler.fit_transform(X_train)
                X_test = ptyj_scaler.fit_transform(X_test)
            if scaling == "PTBC_scale":
                ptbc_scaler = PowerTransformer(method='box-cox')
                X_train = ptbc_scaler.fit_transform(X_train)
                X_test = ptbc_scaler.fit_transform(X_test)
            if scaling == "QTN_scale":
                qtn_scaler = QuantileTransformer(output_distribution='normal')
                X_train = qtn_scaler.fit_transform(X_train)
                X_test = qtn_scaler.fit_transform(X_test)
            if scaling == "QTU_scale":
                qtu_scaler = QuantileTransformer(output_distribution='uniform')
                X_train = qtu_scaler.fit_transform(X_train)
                X_test = qtu_scaler.fit_transform(X_test)
            if norms == "MM_norm":
                X_train = minmax_scale(X_train, feature_range=(0, 1), axis=0, copy=True)
                X_test = minmax_scale(X_test, feature_range=(0, 1), axis=0, copy=True)
            if norms == "M_norm":
                m_norm = Normalizer(norm="max")
                X_train = m_norm.fit_transform(X_train)
                X_test = m_norm.fit_transform(X_test)
            if norms == "LAD_norm":
                lad_norm = Normalizer(norm="l1")
                X_train = lad_norm.fit_transform(X_train)
                X_test = lad_norm.fit_transform(X_test)
            if norms == "LSE_norm":
                lad_norm = Normalizer(norm="l2")
                X_train = lad_norm.fit_transform(X_train)
                X_test = lad_norm.fit_transform(X_test)
            if models == "class_models":
                classifier_lr = LogisticRegression(random_state=0)
                classifier_kNN = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2)
                classifier_SVC = SVC(kernel='linear', random_state=0)
                classifier_kSVC = SVC(kernel='rbf', random_state=0)
                classifier_NB = GaussianNB(priors=None, var_smoothing=1e-09)
                classifier_DT = DecisionTreeClassifier(criterion='entropy', random_state=0)
                classifier_RF = RandomForestClassifier(n_estimators=500, criterion='entropy', bootstrap=True,
                                                       random_state=0, )
                modells = ['classifier_lr', 'classifier_kNN', 'classifier_SVC', 'classifier_kSVC', 'classifier_NB',
                           'classifier_DT', 'classifier_RF']
                dd = []
                for cl in ['lr', 'kNN', 'SVC', 'kSVC', 'NB', 'DT', 'RF']:
                    classifier = eval("classifier_" + cl)
                    classifier.fit(X_train, y_train.values.ravel())
                    y_pred = classifier.predict(X_test)
                    dd.append(accuracy_score(y_test, y_pred))
                for i in range(len(modells)):
                    modells[i]
                    dd[i]
                context['modelnames'] = modells
                context['r2'] = dd

            if models == "regress_models":
                modells = ['regressor_mlr', 'regressor_svr', 'regressor_dt', 'regressor_rf']
                r2 = []
                for model in modells:
                    if model == "regressor_mlr":
                        model = LinearRegression()
                    if model == "regressor_svr":
                        model = SVR(kernel='sigmoid', gamma='scale')
                    if model == "regressor_dt":
                        model = DecisionTreeRegressor(max_leaf_nodes=10, random_state=0)
                    if model == "regressor_rf":
                        model = RandomForestRegressor(n_estimators=500, max_features="log2", random_state=0)
                    model.fit(X_train, y_train.values.ravel())
                    y_pred = model.predict(X_test)
                    r2.append(r2_score(y_test, y_pred))
                for i in range(len(modells)):
                    modells[i]
                    r2[i]
                context['modelnames'] = modells
                context['r2'] = r2
            if len(ycol0) != 1:
                context['selecty'] = 'Please select only one y variable'
            if len(xcols0) < 1:
                context['selecty'] = 'Please select one or more X variables'
            else:
                df = pd.concat([x_cols, y_col], axis=1)
                graph = {}
                for i in df.columns:
                    layout = go.Layout(autosize=False, width=400, height=400,
                                       title=i,
                                       xaxis=dict(title='Value'),
                                       yaxis=dict(title='Count'),
                                       bargap=0.2,
                                       bargroupgap=0.1)
                    data = go.Figure(data=[go.Histogram(x=df[i])], layout=layout)
                    graph[i] = opy.plot(data, include_plotlyjs=False, output_type='div')
                context['graph'] = graph
            # User Selected Features
            context['duplicate_rows'] = duplicate_rows
            context['missing'] = missing
            context['imputing'] = imputing
            context['col_encode_names'] = col_encode_names
            context['col_label_names'] = col_label_names
            context['col_no_label_names'] = col_no_label_names
            context['xcols'] = xcols
            context['ycol'] = ycol
            context['trainingset_s'] = trainingset_s
            context['testset_s'] = testset_s
            context['scaling'] = scaling
            context['norms'] = norms
            context['models'] = models
            # User Obtained Results
            context['category_imputed'] = data_categorical  # Category Imputed DataFrame
            context['x_cols'] = data_numeric_x  # Numerical Imputed On X Selected Features
            context['y_col'] = data_numeric_y  # Numerical Imputed On Y Selected Features
            context['df_X1'] = df_X1  # OneHotEncode On X Features
            context['df_Y1'] = df_Y1  # OneHotEncode On Y Features
            context['df_X2'] = df_X2  # LabelEncode On X Features
            context['df_Y2'] = df_Y2  # LabelEncode On Y Features
            context['df_X3'] = df_X3  # NO_ENCODING On X Features
            context['df_Y3'] = df_Y3  # NO_ENCODING On X Features
            context['ALL_X_DF'] = ALL_X_DF  # All Encoded On X Features
            context['ALL_Y_DF'] = ALL_Y_DF  # All Encoded On Y Features
            context['X_train'] = X_train  # X Training Set
            context['X_test'] = X_test  # X Test Set
            context['y_train'] = y_train  # Y Training Set
            context['y_test'] = y_test  # Y Test Set
        return self.render_to_response(context)


# Profile
@login_required
def profile(request):
    if request.method == 'POST':
        uu_form = UserUpdateForm(request.POST, instance=request.user)
        pp_form = ProfileUpdateForm(request.POST, request.FILES, instance=request.user.profile)
        if uu_form.is_valid() and pp_form.is_valid():
            uu_form.save()
            pp_form.save()
            messages.success(request, f'Your account has been updated')
            return redirect('dash-profile')
    else:
        uu_form = UserUpdateForm(instance=request.user)
        pp_form = ProfileUpdateForm(instance=request.user.profile)
    context = {
        'uuu_form': uu_form,
        'pup_form': pp_form
    }
    return render(request, 'dash/profile.html', context)


@login_required
def change_password(request):
    if request.method == 'POST':
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)  # Important!
            messages.success(request, f'Your password was successfully updated!')
            return redirect('dash-profile')
    else:
        form = PasswordChangeForm(request.user)
    return render(request, 'dash/password.html', {
        'form': form
    })
