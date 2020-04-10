import os


def list_processor(request):
    listfiles = os.listdir('Media/csvfiles')
    return {'listfiles': listfiles}
