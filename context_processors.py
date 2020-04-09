import os


def list_processor(request):
    listfiles = os.listdir('Media/CVS_FOLDER/')
    return {'listfiles': listfiles}
