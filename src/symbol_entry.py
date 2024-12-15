from .parameter import FunctionParameter


class SymbolEntry:
    def __init__(
        self,
        in_object,
        in_name: str | None = None,
        in_type: str | None = None,
        in_const: bool | None = None,
        in_array: bool = False,
    ) -> None:
        self.object = in_object
        if in_name is None:
            self.name = self.object.key
        else:
            self.name = in_name
        if in_const is None:
            self.const = self.object.const
        else:
            self.const = in_const
        if in_type is None:
            self.type = self.object.type
        else:
            self.type = in_type

        self.array = self.object.array
        if self.array:
            self.size = self.object.size
        self.register = None
        self.symbol_table = None
        self.used = False
        self.cast = False
        self.returned = False
        self.owner = None

    def initialized(self):
        return self.const is not None and self.type is not None

    def __repr__(self):
        repr = (
            f"{'const ' if self.const else ''}{self.type + ' '}{self.name}"
            f"{'*'*(self.object.total_deref - self.object.deref_level) if self.object.ptr else ''}"
            f"{f'[{self.size}]' if self.array else ''} : "
        )
        if self.object is not None:
            repr += f"{self.object.value if not self.array else self.object.values}"
        return repr

    def __eq__(self, o: object):
        if not isinstance(o, SymbolEntry):
            return False
        return self.name == o.name and self.type == o.type and self.const == o.const

    def __ne__(self, o: object):
        return self.__eq__(o)

    def print(self):
        print(
            "{:<2}{:<8}{:<2}{:<8}{:<2}{:<15}{:<2}{:<12}{:<2}{:<8}{:<2}".format(
                "|", "const", "|", "type", "|", "name", "|", "value", "|", "used", "|"
            )
        )
        const = "const" if self.const else "None"
        if self.object.value is None and not self.array:
            value = "None"
        elif self.array:
            value = self.object.values
        else:
            value = self.object.value
        used = "True" if self.used else "False"
        print(
            "{:<2}{:<8}{:<2}{:<8}{:<2}{:<15}{:<2}{:<12}{:<2}{:<8}{:<2}".format(
                "|", const, "|", self.type, "|", self.name, "|", value, "|", used, "|"
            )
        )


class FuncSymbolEntry(SymbolEntry):
    def __init__(
        self,
        in_object,
        in_name: str | None = None,
        in_type: str | None = None,
        in_const: bool | None = None,
        in_parameters: list[FunctionParameter] | None = None,
    ) -> None:
        super().__init__(in_object, in_name, in_type, in_const)
        self.parameters = in_parameters
        if self.parameters is None:
            self.parameters = []
        self.defined = False

    def __repr__(self):
        return f"{'const ' if self.const else ''}{self.type} {self.name}"

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, FuncSymbolEntry):
            return False
        if super().__eq__(o):
            if len(self.parameters) != len(o.parameters):
                return False
            for i in range(len(self.parameters)):
                if self.parameters[i] != o.parameters[i]:
                    return False
            return True
        return False

    def __ne__(self, o: object) -> bool:
        return not self.__eq__(o)

    def get_str(self):
        out = ""
        for par in self.parameters:
            out += par.get_str()
            if self.parameters.index(par) != len(self.parameters) - 1:
                out += ","
        return out


class VarSymbolEntry(SymbolEntry):
    def __init__(
        self,
        in_object,
        in_name: str,
        in_type: str,
        in_const: bool | None = None,
        in_ptr: bool | None = None,
    ) -> None:
        super().__init__(in_object, in_name, in_type, in_const)
        self.is_ptr = in_ptr

    def __eq__(self, o: object):
        if not isinstance(o, VarSymbolEntry):
            return False
        return super().__eq__(o) and (self.is_ptr and o.is_ptr)

    def __ne__(self, o: object):
        return super().__ne__(o)


class PtrVarEntry(VarSymbolEntry):
    def __init__(
        self,
        in_object,
        in_name: str,
        in_type: str,
        in_const: bool = False,
        in_ptr: bool = False,
        prev_level=None,
        next_level=None,
    ) -> None:
        super().__init__(in_object, in_name, in_type, in_const, in_ptr)
        self.prev_level = prev_level
        self.next_level = next_level

    def __eq__(self, o: object):
        if not isinstance(o, PtrVarEntry):
            return False
        return (
            super().__eq__(o)
            and self.prev_level == o.prev_level
            and self.next_level == o.next_level
        )

    def __ne__(self, o):
        return not self.__ne__(o)
