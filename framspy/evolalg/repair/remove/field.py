from evolalg.repair.remove.remove import Remove


class FieldRemove(Remove):
    def __init__(self, field_name, field_value, *args, **kwargs):
        super(FieldRemove, self).__init__(*args, **kwargs)
        self.field_name = field_name
        self.field_value = field_value

    def remove(self, individual):
        return not hasattr(individual, self.field_name) or getattr(individual, self.field_name) == self.field_value

