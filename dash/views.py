import os
import plotly.graph_objs as go
import plotly.offline as opy
from django.views.generic import TemplateView
from dash.models import DownloadedFile, CurrentFile,Prepross
from django.shortcuts import render, redirect, reverse
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib.auth.decorators import login_required
from .forms import UserUpdateForm, ProfileUpdateForm
from django.contrib import messages
from django.shortcuts import render
import pandas as pd


class upload(TemplateView):
    template_name = 'dash/upload.html'

    def post(self, request, **kwargs):
        if request.method == 'POST' and request.FILES["docfile"]:

            downloaded_file = DownloadedFile(docfile=request.FILES['docfile'])
            downloaded_file.save()
            downloaded_file.delete() # it deletes only from database
            return render(request,'dash/upload.html')

    def __str__(self):
        return self.name


class index(TemplateView):
    template_name = 'dash/index.html'

    def post(self, request, **kwargs):
        if request.method == 'POST':
            current_file = CurrentFile(filename = list(request.POST.keys())[1])
            current_file.save()
            context = {}
            #listfiles = os.listdir('media/downloaded')
            context['fileselected'] = list(request.POST.keys())[1]
            context['test'] = 'a'
            #context['listfiles'] = listfiles
            return render(request,'dash/index.html', context)

    def __str__(self):
        return self.name

# *****************************DATA PREPROCESSING*********************************************

class prepross(TemplateView):
    template_name = 'dash/preprocessing.html'
    #model = Prepross
    #fields = ('file_name','coltype','Xvars','Yvar','onehot','featscaling','na_omit','ordinal')

    def get_context_data(self, **kwargs):
        data = super().get_context_data(**kwargs)
        context = {}
        try:
            # *****************************LOADING DATA FROM MODELS*************************************
            file_name = CurrentFile.objects.order_by('-id')[0].filename #it gets the uploaded filename from database
            df = pd.read_csv(os.path.join('media\downloaded', file_name)) # it displays uploaded dataset from database
            count_nan = len(df) - df.count() # it counts the number of missing values
            row_count = df.count()[0] #it counts the number of rows in the dataset
            file_type = pd.concat([df.dtypes, count_nan, df.nunique()], axis=1) #concat the total dataset https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html
            file_type.columns = ("Type", "NA count", "Count distinct")#displaying details of the dataset

            context['file_type'] = file_type
            context['row_count'] = row_count
        except:
            file_name = 'Please select one file'


        context['file_name'] = file_name
        return context

    def post(self, request, **kwargs):
        if request.method == 'POST': #post the requests

            # Entry to model of the data preprocessing to post the requests
            # *****************************LOADING DATA FROM MODELS*************************************
            prep_file = Prepross.objects.get_or_create(filename = CurrentFile.objects.order_by('-id')[0].filename,
                                  coltype = request.POST.getlist('coltype'),
                                    assvar = request.POST.getlist('assvar'),
                            missingvalues = request.POST.getlist('missingvalues'),
                            trainingset_size = request.POST['trainingset_size'],
                              featscaling = request.POST.getlist('featscaling'),
                               ordinal =  request.POST.getlist('ordinal'),

                                   )

            # Get dataframe and change data type
            #https://docs.python.org/3/library/functions.html
            # i used python builtin function to write this function
            context = {}
            file_name = CurrentFile.objects.order_by('-id')[0].filename
            coltype = request.POST.getlist('coltype')
            coltype = dict([i.split(':', 1)for i in coltype])#which spilts the columns
            df = pd.read_csv(os.path.join('dash\CSV_FOLDER', file_name), dtype= coltype)
            row_count = df.count()[0]

            # Keep only selected columns
            assvar = request.POST.getlist('assvar')
            xcols0 = [s for s in assvar if ":X" in s]# selected x columns
            xcols = [i.split(':', 1)[0] for i in xcols0]
            ycol0 = [s for s in assvar if ":y" in s]
            ycol = [i.split(':', 1)[0] for i in ycol0]#selected y columns
            cols = xcols + ycol
            df = df[cols]

            xcols = ', '.join(xcols)#joining the x columns
            ycol = ', '.join(ycol)#joining the y columns
            missing = request.POST.getlist('missingvalues')#posting the missing values
            missing = ', '.join(missing)#join()accept any iterable
            trainingset_s = request.POST.getlist('trainingset_size')
            trainingset_s = ', '.join(trainingset_s)
            testset_s = 100 - int(trainingset_s)
            feat =  request.POST['featscaling']
            encode = request.POST['ordinal']

            # Taking care of missing data in the data set
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

                # Ploting thgraphs by using plotly
                #https://plot.ly/

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
            #displaying the selected data
            context['xcols'] = xcols
            context['ycol'] = ycol
            context['missing'] = missing
            context['trainingset_s'] = trainingset_s
            context['testset_s'] = testset_s
            context['feat'] = feat
            context['encode'] = encode
            context['file_name'] = file_name
            context['row_count'] = row_count
            return render(request,'dash/preprocessing.html', context)

    def __str__(self):
        return self.name


















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

