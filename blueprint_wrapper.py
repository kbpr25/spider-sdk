from flask import Blueprint as FlaskBlueprint

class Blueprint(FlaskBlueprint):
    def __init__(self, name, *args, **kwargs):
        if '.' in name:
            raise ValueError('Blueprint name cannot contain dots')
        super().__init__(name, *args, **kwargs)