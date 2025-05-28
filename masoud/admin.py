from django.contrib import admin

# Register your models here.
from import_export.admin import ImportExportModelAdmin
from .models import Parts

@admin.register(Parts)
class Train_info_Ressources(ImportExportModelAdmin):
    class Meta:
        model = Parts


## password : ****