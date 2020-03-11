from import_export import resources, fields, widgets
from import_export.admin import ImportExportModelAdmin
from django.contrib import admin
from .models import Profile


class ProfileAdmin(admin.ModelAdmin):
    list_display = {'title', 'date_created', 'last_modified', 'is_draft'}


admin.site.register(Profile)



