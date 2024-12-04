from .ast import FuncParameter


class FunctionParameter:
    def __init__(self, in_object: FuncParameter) -> None:
        self.type = in_object.type
        self.ptr = in_object.ptr
        self.ptr_level = in_object.total_deref - in_object.deref_level
        self.reference = in_object.reference
        self.array = in_object.array
        self.const = in_object.const
        self.name = in_object.key
        self.default_value = in_object.value
        self.object = in_object

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, FunctionParameter):
            return False
        return self.type == o.type and self.name == o.name and self.default_value == o.default_value

    def __ne__(self, o: object) -> bool:
        return not self.__eq__(o)

    def get_str(self):
        return f"{self.type} {self.name} = {self.default_value} "

    def __repr__(self):
        return f"{self.type} {self.name} = {self.default_value}"

    def update(self, register, registers):
        self.object.update(register, registers)