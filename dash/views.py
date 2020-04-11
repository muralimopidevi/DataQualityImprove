import os
import plotly.graph_objs as go
import plotly.offline as opy
from dash.models import CurrentFile, Prepross
from django.shortcuts import redirect
from django.views.generic import TemplateView, CreateView
from django.urls import reverse_lazy
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib.auth.decorators import login_required
from .forms import UserUpdateForm, ProfileUpdateForm
from django.contrib import messages
from django.shortcuts import render
import pandas as pd
from .forms import CSVForm
from .models import CSV


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
        if request.method == 'POST':
            current_file = CurrentFile(filename=list(request.POST.keys())[1])
            current_file.save()
            context = {}
            context['fileselected'] = list(request.POST.keys())[1]
            context['test'] = 'a'
            return render(request, 'dash/index.html', context)

    def __str__(self):
        return self.name


# ------------------ PREPROCESSING  ------------------------- #
class prepross(TemplateView):
    template_name = 'dash/preprocessing.html'
    # model = Prepross
    # fields = ('file_name','coltype','Xvars','Yvar','onehot','featscaling','na_omit')

    def get_context_data(self, **kwargs):
        data = super().get_context_data(**kwargs)
        context = {}
        try:
            file_name = CurrentFile.objects.order_by('-id')[0].filename
            df = pd.read_csv(os.path.join('Media\csvfiles', file_name))
            count_nan = len(df) - df.count()
            row_count = df.count()[0]
            file_type = pd.concat([df.dtypes, count_nan, df.nunique()], axis=1)
            file_type.columns = ("Type", "NA count", "Count distinct")
            context['df'] = df
            context['vals'] = df.values.tolist()
            context['file_type'] = file_type
            context['row_count'] = row_count
        except:
            file_name = 'Please select one file'

        context['file_name'] = file_name
        return context

    def post(self, request, **kwargs):
        if request.method == 'POST':

            # Entry to model
            prep_file = Prepross.objects.get_or_create(filename=CurrentFile.objects.order_by('-id')[0].filename,
                                                       coltype=request.POST.getlist('coltype'),
                                                       assvar=request.POST.getlist('assvar'),
                                                       missingvalues=request.POST.getlist('missingvalues'),
                                                       trainingset_size=request.POST['trainingset_size'],
                                                       featscaling=request.POST.getlist('featscaling'))

            # Get dataframe and change data type
            context = {}
            file_name = CurrentFile.objects.order_by('-id')[0].filename
            coltype = request.POST.getlist('coltype')
            coltype = dict([i.split(':', 1) for i in coltype])
            df = pd.read_csv(os.path.join('Media\csvfiles', file_name), dtype= coltype)
            row_count = df.count()[0]

            # Keep only selected columns
            assvar = request.POST.getlist('assvar')
            xcols0 = [s for s in assvar if ":X" in s]
            xcols = [i.split(':', 1)[0] for i in xcols0]
            ycol0 = [s for s in assvar if ":y" in s]
            ycol = [i.split(':', 1)[0] for i in ycol0]
            cols = xcols + ycol
            df = df[cols]

            xcols = ', '.join(xcols)
            ycol = ', '.join(ycol)
            missing = request.POST.getlist('missingvalues')
            missing = ', '.join(missing)
            trainingset_s = request.POST.getlist('trainingset_size')
            trainingset_s = ', '.join(trainingset_s)
            testset_s = 100 - int(trainingset_s)
            feat =  request.POST['featscaling']

            # Taking care of missing data
            if missing == "no":
                if len(df) != len(df.dropna()):
                    context['selecty'] = 'Your data seem to have Missing Values'
                else:
                    df = df.dropna()

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
            context['trainingset_s'] = trainingset_s
            context['testset_s'] = testset_s
            context['feat'] = feat
            context['file_name'] = file_name
            context['row_count'] = row_count
            return render(request, 'dash/preprocessing.html', context)

    def __str__(self):
        return self.name


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
