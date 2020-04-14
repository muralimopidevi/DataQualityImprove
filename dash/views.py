import os
import pandas as pd
import numpy as np
from django.contrib import messages
from django.shortcuts import render
from django.shortcuts import redirect
from django.views.generic import TemplateView, CreateView
from django.urls import reverse_lazy
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib.auth.decorators import login_required
import plotly.graph_objs as go
import plotly.offline as opy
from sklearn.impute import SimpleImputer, KNNImputer
from missingpy import MissForest
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
class prepross(TemplateView):
    template_name = 'dash/preprocessing.html'
    # model = Prepross
    # fields = ('file_name','coltype','Xvars','Yvar','onehot','featscaling','Drop_NA')

    def get_context_data(self, **kwargs):
        data = super().get_context_data(**kwargs)
        context = {}
        try:
            file_name = CurrentFile.objects.order_by('-id')[0].filename
            df1 = pd.read_csv(os.path.join('Media\csvfiles', file_name))
            # Infer Data types
            df = df1.infer_objects()
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
        if request.method == 'POST':
            # Get dataframe and change data type
            context = {}
            file_name = CurrentFile.objects.order_by('-id')[0].filename
            coltype = request.POST.getlist('coltype')
            coltype = dict([i.split(':', 1) for i in coltype])
            df = pd.read_csv(os.path.join('Media\csvfiles', file_name), low_memory=False, dtype=coltype)
            row_count = df.count()[0]
            # Keep only selected columns
            assvar = request.POST.getlist('assvar')
            xcols0 = [s for s in assvar if ":X" in s]
            xcols = [i.split(':', 1)[0] for i in xcols0]  # Select X cols from user form
            ycol0 = [s for s in assvar if ":y" in s]
            ycol = [i.split(':', 1)[0] for i in ycol0]  # Select y cols from user form
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
                    context['selecty'] = 'Your data seem to have Missing Values'
            if missing == "Drop_NAN":
                df = df.dropna()
            if imputing == "mean" or\
                    imputing == "median" or\
                    imputing == "most_frequent" or\
                    imputing == "constant":
                impute = imputing
                categorical_columns = []
                numeric_columns = []
                for c in df.columns:
                    if df[c].map(type).eq(str).any():  # check if there are any strings in column
                        categorical_columns.append(c)
                    else:
                        numeric_columns.append(c)
                # create two DataFrames, one for each data type
                data_numeric = df[numeric_columns]
                data_categorical = pd.DataFrame(df[categorical_columns])
                imp = SimpleImputer(missing_values=np.nan, strategy=impute)
                data_numeric = pd.DataFrame(imp.fit_transform(data_numeric),
                                            columns=data_numeric.columns)  # only apply imputer to numeric columns
                # you could do something like one-hot-encoding of data_categorical here
                # join the two masked data frames back together
                df = pd.concat([data_numeric, data_categorical], axis=1)
            if imputing == "KNN":
                categorical_columns = []
                numeric_columns = []
                for c in df.columns:
                    if df[c].map(type).eq(str).any():  # check if there are any strings in column
                        categorical_columns.append(c)
                    else:
                        numeric_columns.append(c)
                # create two DataFrames, one for each data type
                data_numeric = df[numeric_columns]
                data_categorical = pd.DataFrame(df[categorical_columns])
                imp = KNNImputer(n_neighbors=5, weights="uniform")
                data_numeric = pd.DataFrame(imp.fit_transform(data_numeric),
                                            columns=data_numeric.columns)  # only apply imputer to numeric columns
                # you could do something like one-hot-encoding of data_categorical here
                # join the two masked data frames back together
                df = pd.concat([data_numeric, data_categorical], axis=1)
            if imputing == "RDF":
                categorical_columns = []
                numeric_columns = []
                for c in df.columns:
                    if df[c].map(type).eq(str).any():  # check if there are any strings in column
                        categorical_columns.append(c)
                    else:
                        numeric_columns.append(c)
                # create two DataFrames, one for each data type
                data_numeric = df[numeric_columns]
                data_categorical = pd.DataFrame(df[categorical_columns])
                imp = MissForest(max_iter=10, decreasing=False, missing_values=np.nan,
                                 copy=True, n_estimators=100, criterion=('mse', 'gini'),
                                 max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                 min_weight_fraction_leaf=0.0, max_features='auto',
                                 max_leaf_nodes=None, min_impurity_decrease=0.0,
                                 bootstrap=True, oob_score=False, n_jobs=-1, random_state=None,
                                 verbose=0, warm_start=False, class_weight=None)
                data_numeric = pd.DataFrame(imp.fit_transform(data_numeric),
                                            columns=data_numeric.columns)  # only apply imputer to numeric columns
                # you could do something like one-hot-encoding of data_categorical here
                # join the two masked data frames back together
                df = pd.concat([data_numeric, data_categorical], axis=1)
            # Return error if columns are not selected
            if len(ycol0) != 1:
                context['selecty'] = 'Please select one y variable'
            elif len(xcols0) < 1:
                context['selecty'] = 'Please select one or more X variables'
            else:
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
            context['xcols'] = xcols
            context['ycol'] = ycol
            context['missing'] = missing
            context['imputing'] = imputing
            context['trainingset_s'] = trainingset_s
            context['testset_s'] = testset_s
            context['file_name'] = file_name
            context['row_count'] = row_count
            context['df'] = df

            return render(request, 'dash/preprocessing.html', context)

    def __str__(self):
        return self.template_name


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