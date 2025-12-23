import os
from django.db.models.fields import FilePathField

class CallableFilePathField(FilePathField):
    """
    FilePathField that accepts a callable for the path parameter.
    The callable will be evaluated at runtime rather than during migrations.
    """
    def __init__(self, path='', *args, **kwargs):
        self._path_callable = path if callable(path) else None
        initial_path = path() if callable(path) else path
        super().__init__(path=initial_path, *args, **kwargs)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if self._path_callable:
            # For migrations, we need to represent the callable in a serializable way
            # This assumes the callable is a named function or method
            kwargs['path'] = f'{self._path_callable.__module__}.{self._path_callable.__qualname__}'
        return name, path, args, kwargs

    def get_prep_value(self, value):
        if self._path_callable:
            self.path = self._path_callable()
        return super().get_prep_value(value)