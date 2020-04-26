import os
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.offline as opy
from collections import defaultdict
from scipy.sparse import csr_matrix
from django.contrib import messages
from django.shortcuts import render
from django.shortcuts import redirect
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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
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

    def __str__(self):
        return self.template_name


# ------------------ PREPROCESSING  ------------------------- #
class prepross(TemplateView, TransformerMixin, ContextMixin):
    template_name = 'dash/preprocessing.html'

    # model = Prepross
    # fields = ('file_name','coltype','Xvars','Yvar','onehot','featscaling','Drop_NA')
    def __init__(self):
        """Impute missing values.
        Columns of dtype object are imputed with the most frequent value
        in column.
        Columns of other types are imputed with mean of column.
        """
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
            df1 = pd.read_csv(os.path.join('Media\csvfiles', file_name), low_memory=False, delimiter=',')
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
        global df_X1, df_X2, df_X3, df_Y1, df_Y2, df_Y3, ALL_Y_DF, ALL_X_DF
        df_X1 = pd.DataFrame()
        df_X2 = pd.DataFrame()
        df_X3 = pd.DataFrame()
        df_Y1 = pd.DataFrame()
        df_Y2 = pd.DataFrame()
        df_Y3 = pd.DataFrame()
        context = self.get_context_data(**kwargs)
        if request.method == 'POST':
            # Get dataframe and change data type
            file_name = CurrentFile.objects.order_by('-id')[0].filename
            coltype = request.POST.getlist('coltype')
            coltype = dict([i.split(':', 1) for i in coltype])
            # Reading Dataframe
            df = pd.read_csv(os.path.join('Media\csvfiles', file_name), low_memory=False, delimiter=',', dtype=coltype)

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
            row_count = df.count()[0]

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
            missing = request.POST.getlist('missingvalues')
            missing = ', '.join(missing)
            imputing = request.POST.getlist('imputing')
            imputing = ', '.join(imputing)
            trainingset_s = request.POST.getlist('trainingset_size')
            trainingset_s = ', '.join(trainingset_s)
            testset_s = 100 - int(trainingset_s)
            # Taking care of missing data
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
            if any([item in x_cols for item in col_encode_names_list]):
                A = list(x_cols)
                B = col_encode_names_list
                C = list(set(A) & set(B))
                col_encode_names_vals_x = pd.DataFrame(x_cols[C])
                ohe = OneHotEncoder()
                ct = make_column_transformer((ohe, C), remainder='passthrough')
                ctt = ct.fit_transform(col_encode_names_vals_x)
                df_X1 = pd.DataFrame.sparse.from_spmatrix(ctt)
                # print("column transformer - X")
                # print(df_X1)

            if any([item in y_col for item in col_encode_names_list]):
                A = list(y_col)
                B = col_encode_names_list
                C = list(set(A) & set(B))
                col_encode_names_vals_y = pd.DataFrame(y_col[C])
                ohe = OneHotEncoder()
                ct = make_column_transformer((ohe, C), remainder='passthrough')
                ctt = ct.fit_transform(col_encode_names_vals_y)
                df_Y1 = pd.DataFrame.sparse.from_spmatrix(ctt)
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
            print(X_train)
            print(X_test)
            print(y_train)
            print(y_test)
            context['x_cols'] = ALL_X_DF
            context['y_col'] = ALL_Y_DF
            context['xcols'] = xcols
            context['ycol'] = ycol
            context['col_encode_names'] = col_encode_names
            context['col_label_names'] = col_label_names
            context['missing'] = missing
            context['imputing'] = imputing
            context['trainingset_s'] = trainingset_s
            context['testset_s'] = testset_s
            context['file_name'] = file_name
            context['row_count'] = row_count
            context['df'] = df
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
