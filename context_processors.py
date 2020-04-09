import os


def list_processor(request):
    listfiles = os.listdir('Media/csv')
    return {'listfiles': listfiles}
