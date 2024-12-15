from array import array
from math import floor


class Node:
    def __init__(self, key: str, value, parent=None) -> None:
        """
        Initializer function
        :param key: key value of the node
        :param value: value of the node
        """
        super().__init__()
        self.key = key
        self.value = value
        self.parent = parent
        self.cast = False
        self.register = None
        self.in_loop = False
        self.in_func = False
        self.type = None
        self.addr = False
        self.deref = False
        self.known = True

    # def __eq__(self, o: object) -> bool:
    #     return (self.key == o.key) and (self.value == o.value)
    #
    # def __ne__(self, o: object) -> bool:
    #     return not self.__eq__(o)

    def __repr__(self) -> str:
        return f"{self.key} : {self.value}"

    def print(self):
        """
        print function of node
        :return: None
        """
        print(self.get_str())

    def __mul__(self, other):
        if isinstance(self.value, str):
            self.value = ord(self.value)
        if isinstance(other.value, str):
            other.value = ord(other.value)
        return Node("", self.value * other.value)

    def __floordiv__(self, other):
        if other.value != 0:
            if isinstance(self.value, str):
                self.value = ord(self.value)
            if isinstance(other.value, str):
                other.value = ord(other.value)
            return Node("int", floor(self.value / other.value))
        raise ZeroDivisionError

    def __truediv__(self, other):
        if other.value != 0:
            if isinstance(self.value, str):
                self.value = ord(self.value)
            if isinstance(other.value, str):
                other.value = ord(other.value)
            return Node(
                f"{'float' if not (isinstance(self.value, int) or isinstance(other.value, int)) else 'int'}",
                self.value / other.value,
            )
        raise ZeroDivisionError

    def __add__(self, other):
        if isinstance(self.value, str):
            self.value = ord(self.value)
        if isinstance(other.value, str):
            other.value = ord(other.value)
        if isinstance(other, int | float):
            return Node("", self.value + other)
        return Node("", self.value + other.value)

    def __sub__(self, other):
        if isinstance(self.value, str):
            self.value = ord(self.value)
        if isinstance(other.value, str):
            other.value = ord(other.value)
        if isinstance(other, int | float):
            return Node("", self.value - other)
        return Node("", self.value - other.value)

    def __mod__(self, other):
        if isinstance(self.value, str):
            self.value = ord(self.value)
        if isinstance(other.value, str):
            other.value = ord(other.value)
        return Node("", self.value % other.value)

    def __neg__(self):
        if isinstance(self.value, str):
            self.value = ord(self.value)
        return Node("", -self.value)

    def __lt__(self, other):
        if isinstance(self.value, str):
            self.value = ord(self.value)
        if isinstance(other.value, str):
            other.value = ord(other.value)
        return Node("int", self.value < other.value)

    def __le__(self, other):
        if isinstance(self.value, str):
            self.value = ord(self.value)
        if isinstance(other.value, str):
            other.value = ord(other.value)
        return Node("int", self.value <= other.value)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        # if isinstance(self.value, str) and self.key != "var" and len(self.value) == 1:
        #     self.value = ord(self.value)
        # if isinstance(self.value, str) and self.key != "var":
        #     self.value = ord(self.value)
        # if isinstance(other.value, str) and other.key != "var":
        #     other.value = ord(other.value)
        if not isinstance(self, VarNode) and not isinstance(other, VarNode):
            return self.value == other.value and self.key == other.key
        return self.value == other.value

    def __ne__(self, other):
        return not self.__eq__(other)

    def __gt__(self, other):
        if isinstance(self.value, str):
            self.value = ord(self.value)
        if isinstance(other.value, str):
            other.value = ord(other.value)
        return Node("int", self.value > other.value)

    def __ge__(self, other):
        if isinstance(self.value, str):
            self.value = ord(self.value)
        if isinstance(other.value, str):
            other.value = ord(other.value)
        return Node("int", self.value >= other.value)

    def save(self):
        """
        Saves the node in a json / dictionary format
        :return: dictionary
        """
        # check if parent is an array, thus this node is an element of an array
        if isinstance(self.parent, ArrayNode):
            index = self.parent.values.index(self)
            out_key = f"{self.parent.get_str()}[{index}]"  # key of the node
            return {out_key: self.value}
        if isinstance(self.value, VarNode):
            return {self.key: self.value.save()}
        return {self.key: self.value}

    def get_str(self):
        """
        converts Node in a string format
        :return:
        """
        return f"{self.key}\t:\t{self.value!s}"

    def save_dot(self):
        """
        converts Node in a dot dictionary format
        :return: dot dictionary format
        """
        return f"\"{self.key}\" [label=\"{self.key}{ ' :' + self.value if self.value is not None else ''}\"];\n"

    def recursive_dot(self, dictionary, count, name):
        """
        recursion for the dot format
        :return: None
        """
        if self.key not in dictionary or count[self.key] == 1:
            dictionary[self.key] = set()
            name = self.key
        else:
            dictionary[name] = set()

        if isinstance(self.value, Node):
            dictionary[name].add(self.value.key)
            self.value.recursive_dot(dictionary, count, self.value.key)
        else:
            dictionary[name].add(self.value)

    def llvm(self, scope: bool = False, index: int = 0) -> tuple[str, int]:
        return " ", index

    def mips(self, registers):
        out_global = ""
        out_local = ""
        out_list = []
        if self.value is None and self.key != "var":
            return "", "", []
        # check if the value is already in a register
        registers.search(self)
        if self.register is None:
            # load the value in a register
            registers.temporaryManager.LRU(self)
        if self.key == "var":
            # variable is declared in the data section
            if self.type == "int":
                temp_type = "int"
            elif self.type == "float":
                temp_type = "flt"
            elif self.type == "char":
                temp_type = "chr"
            else:
                temp_type = "int"
            out_local += f"\tlw{'c1' if (self.key == 'float' or self.register.name[0] == 'f') else ''} ${self.register.name}, {temp_type}_{self.value}\t# {self.get_str()}"
            out_local += f"\t# {self.get_str()}\n"
            # out_local += f"\tla ${self.register.name}, {self.value}\n"
        else:
            out_local += f"\tli ${self.register.name}, {self.value}\t# {self.get_str()}"
            out_local += f"\t\t# {self.get_str()}\n"
        if isinstance(self.parent, ArrayNode):
            temp_node = Node("", None)
            registers.temporaryManager.LRU(temp_node)
            out_local += f"\tla ${temp_node.register.name}, {self.parent.type}_{self.parent.key}\n"
            out_local += f"\taddi ${temp_node.register.name}, ${temp_node.register.name}, {self.parent.values.index(self) * 4}\n"
            out_local += f"\tsw ${self.register.name}, 0(${temp_node.register.name})\n"
        out_list.append(self.register.name)
        self.register.shuffle()
        # place holder
        return out_local, out_global, out_list

    def update(self, register, registers):
        if registers.search(self) is not None:
            self.register.clear()
            self.register = None
        register.update(self)


class VarNode(Node):
    def __init__(
        self,
        key: str,
        value,
        vtype: str,
        const: bool = False,
        ptr: bool = False,
        deref_level: int = 0,
        total_deref: int = 0,
        const_ptr: bool = False,
        is_array: bool = False,
    ) -> None:
        """
        Initializer function for VarNode
        """
        super().__init__(key, value)
        self.type = vtype
        self.const = const
        self.ptr = ptr
        self.deref_level = deref_level
        self.total_deref = total_deref
        self.const_ptr = const_ptr
        self.array = is_array

    def __repr__(self) -> str:
        return f"{self.type} {'*' * (self.total_deref - self.deref_level - 1)} {self.key} : {self.value}"

    def __eq__(self, o):
        if not isinstance(o, VarNode):
            return False
        return (
            self.key == o.key
            and (self.const == o.const or (self.const is None and o.const is not None))
            and self.ptr == o.ptr
        )

    def __ne__(self, o):
        return not self.__eq__(o)

    def save(self):
        """
        Converts VarNode in a json / dictionary format
        :return: dictionary
        """
        out_key = f"{'const ' if self.const else ''}{self.type}{'*' * (self.total_deref - self.deref_level)} {self.key}"
        if isinstance(self.value, VarNode):
            # out_key = str('*'*(self.deref_level+1)) + out_key
            out = {out_key: self.value.save()}
        else:
            out = {out_key: self.value}
        return out

    def assign_type(self, vtype):
        """
        assign a type to VarNode
        :param vtype: variable type
        :return: None
        """
        self.type = vtype
        if isinstance(self.value, VarNode):
            self.value.assign_type(vtype)

    def get_str(self):
        """
        string format of VarNode
        :return: string
        """
        return f"{'const ' if self.const else ''}{self.type}{'*' * (self.total_deref - self.deref_level)} {self.key} : {self.value}"

    def save_dot(self):
        """
        dot format
        :return: string
        """
        out = '"' + self.type + " " + self.key + '"' + "\t" + "->" + "\t"
        if isinstance(self.value, VarNode):
            out += self.value.save_dot()
        elif isinstance(self.value, str):
            out += '"\\' + self.value + '"'
        else:
            out += str(self.value)
        return out

    def llvm(self, scope: bool = False, index: int = 0) -> tuple[str, int]:
        """
        converts VarNode in to llvm
        :param scope: global, local --> True is local
        :return: str
        """
        """
        @y = global i32 4, align 4
        @x = global ptr @y, align 8
        @z = global ptr @x, align 8
        """

        if self.value is None:
            if self.type == "int":
                self.value = 0
            elif self.type == "float":
                self.value = 0.0
            elif self.type == "char":
                self.value = ord("\0")
        # Scope and constant
        out = f"%{index} = " if scope and not self.const else f"@{index} = "

        if self.const:
            out += "constant "
        elif not scope:
            out += "global "
        elif scope:
            out += "alloca "
        var_type = ""
        # type
        if not self.ptr:
            if self.type == "int":
                var_type = "i32"
                out += "i32 "
            elif self.type == "char":
                var_type = "i32"
                out += "i8 "
            elif self.type == "float":
                var_type = "i32"
                out += "float "
            out += " , align 4\n"
        else:
            out += f"ptr @{index}, align 8\n"
        # align
        out_val = ""
        # convert the value if needed
        if not self.ptr:
            if self.type == "float" and not self.ptr:
                if self.value is None:
                    self.value = 0.0
                val = array("f", [self.value])
                self.value = val[0]
                out_val = str(val[0])
            elif isinstance(self.value, str) and not self.ptr:
                if self.value is None:
                    self.value = "\0"
                out_val = str(ord(self.value))
            else:
                if self.value is None:
                    self.value = 0
                out_val = self.value

        # allocate the value
        if not scope:
            out += f"{out_val if not self.ptr else ''} , align {'4' if not self.ptr else '8'}\n"
        else:
            if scope and not self.const:
                out += f"store {var_type} {out_val}, ptr %{index}, align 4\n"
            else:
                if isinstance(self.value, float):
                    val = array("f", [self.value])
                    self.value = val[0]
                    out += f"{out_val}\n"
                elif isinstance(self.value, str):
                    out += str(ord(self.value))
                elif isinstance(self.value, VarNode):
                    out += f"@{self.value.key} , align 8"
                else:
                    out += str(self.value) + "\n"
            out += "\n"
        self.register = index
        return out, index + 1

    def mips(self, registers):
        if self.ptr and isinstance(self.value, VarNode) and self.value.register is None:
            if self.value.const:
                registers.savedManager.LRU(self.value)
            elif self.value.type == "float":
                registers.floatManager.LRU(self.value)
            else:
                registers.temporaryManager.LRU(self.value)
        # assign itself to a register
        # search for a register first
        if self.register is None:
            if registers.search(self) is not None:
                pass
            elif self.const:
                registers.savedManager.LRU(self)
                # registers.globalObjects.data[0][self.value] = self.key
            elif self.type == "float":
                registers.floatManager.LRU(self)
                # registers.globalObjects.data[1][self.value] = self.key
            else:
                registers.temporaryManager.LRU(self)
        if self.value is None:
            return "", "", []
        # mips variable declaration
        # if self.const, also declare in .data
        # get right type
        if self.type == "int":
            out_type = ".word"
        elif self.type == "float":
            out_type = ".float"
        elif self.type == "char":
            out_type = ".byte"
        else:
            out_type = ".word"

        # get right value
        out_val = f"'{self.value}'" if isinstance(self.value, str) else f"{self.value}"

        # if self.const or self.type == "float":
        #     out_global = f"{self.key}: {out_type} {out_val}\n"
        # # if not self.const, declare in .text
        # else:
        out_global = ""
        out_local = ""
        out_reg = []
        # local variable declaration
        if not (self.ptr and isinstance(self.value, VarNode)):
            if self.known:
                if self.type == "float":
                    out_local = f"\tlwc1 ${self.register.name}, flt_{self.key}"
                else:
                    out_local = f"\tli ${self.register.name}, {out_val}"
            else:
                out_type = (
                    "int"
                    if self.type == "int"
                    else "float"
                    if self.type == "flt"
                    else "chr"
                )
                out_local += f"\tlw{'c1' if self.type == 'float' else ''} ${self.register.name}, {out_type}_{self.key}\n"
            self.register.shuffle()
            out_local += f"\t\t# {self.get_str()}\n"
        else:
            # Load the pointed value into the register
            # if self.register is None:
            output = self.value.mips(registers)
            out_local += output[0]
            out_global += output[1]
            out_reg += output[2]
            # load the address of the variable
            out_local += f"\tla ${self.register.name}, {self.value.key if not self.ptr else f'0(${self.value.register.name})'}"
            out_local += f"\t# {self.get_str()}\n"
            self.register.shuffle()
            # store the address in the register
            # out_local += f"\tsw{'c1' if self.type == 'float' else ''} ${self.value.register.name}, 0(${self.register.name})\n"
        if self.register is not None:
            out_reg.append(self.register.name)
        return out_local, out_global, out_reg


class ArrayNode(VarNode):
    def __init__(
        self,
        key: str,
        value,
        vtype: str,
        const: bool = False,
        ptr: bool = False,
        deref_level: int = 0,
        total_deref: int = 0,
        const_ptr: bool = False,
        is_array: bool = True,
        in_size: int = 0,
        in_values=None,
    ) -> None:
        super().__init__(
            key, value, vtype, const, ptr, deref_level, total_deref, const_ptr, is_array
        )
        if in_values is None:
            in_values = []
        self.values: list[VarNode | Node] = []
        self.size = in_size

    def __repr__(self) -> str:
        return (
            f"{self.type} {'*' * self.total_deref} {self.key} [{self.size if self.size > 0 else ''}] : "
            f"[{', '.join([v.value for v in self.values])}]"
            if self.values
            else f"{super().__repr__()}"
        )

    def save(self):
        out_key = (
            f"{'const ' if self.const else ''}{self.type}{'*' * (self.total_deref - self.deref_level)}"
            f" {self.key}[{self.size if self.size > 0 else ''}]"
        )
        return {out_key: [value.save() for value in self.values]}

    def get_str(self):
        # for example: out_key = "int*[5] a"
        return f"{'const ' if self.const else ''}{self.type}{'*' * (self.total_deref - self.deref_level)} {self.key}"

    def __eq__(self, o):
        return (
            isinstance(o, ArrayNode)
            and self.key == o.key
            and self.type == o.type
            and self.const == o.const
            and self.ptr == o.ptr
            and self.deref_level == o.deref_level
            and self.total_deref == o.total_deref
            and self.const_ptr == o.const_ptr
            and self.size == o.size
        )

    def __ne__(self, o):
        return not self.__eq__(o)


class FuncParameter(VarNode):
    def __init__(
        self,
        key: str,
        value,
        vtype: str,
        const: bool = False,
        ptr: bool = False,
        deref_level: int = 0,
        total_deref: int = 0,
        const_ptr: bool = False,
        reference: bool = False,
        is_array=False,
    ) -> None:
        super().__init__(
            key, value, vtype, const, ptr, deref_level, total_deref, const_ptr, is_array
        )
        self.reference = reference
        # if it is a pointer, create a new variable for each deref
        if self.ptr:
            new_values = []
            for i in range(self.total_deref):
                new_val = VarNode(
                    f"{self.key}_{i}",
                    self.value,
                    self.type,
                    self.const,
                    self.ptr,
                    i,
                    self.total_deref,
                    self.const_ptr,
                    self.array,
                )
                new_values.append(new_val)
                if i > 0:
                    new_values[i - 1].value = new_val
                    new_values[i - 1].parent = new_val
                else:
                    new_values[i].parent = self
            self.deref_level = 0
            self.value = new_values[0]

    def __repr__(self) -> str:
        return f"{'& ' if self.reference else ''}{super().__repr__()}"

    def save(self):
        out_key = (
            f"{'const ' if self.const else ''}{self.type}{'*' * (self.total_deref - self.deref_level)}"
            f"{' &' if self.reference else ' '}{self.key}"
        )
        if isinstance(self.value, VarNode):
            out = {out_key: self.value.save()}
        else:
            out = {out_key: self.value}
        return out


class FunctionNode(Node):
    def __init__(
        self, key: str, ret_type: str | None = None, in_const: bool = False
    ) -> None:
        """
        Initializer
        """
        super().__init__(key, None)
        self.body = None
        self.type = ret_type
        self.const = in_const

    def print(self):
        """
        print function fof FunctionNode
        :return: None
        """
        print(self.get_str())

    def save(self):
        """
        convert FunctionNode in to json / dictionary
        :return: dictionary
        """
        if isinstance(self.value, VarNode):
            return {self.key: self.value.save()}
        values = []
        for key, val in self.value.items():
            values.append(str(key) + "=" + str(val.value))
        return {self.key: values}

    def save_dot(self):
        """
        converts FunctionNode in to dot format as dictionary
        :return: dictionary
        """
        if isinstance(self.value, VarNode):
            return {self.key: self.value.save()}
        values = []
        for key, val in self.value.items():
            values.append(str(key) + "=" + str(val.value))
        return {self.key: values}

    def get_str(self):
        """
        string version of FunctionNode
        :return: string
        """
        out = self.key + "\t" + ":" + "\t"
        for key, val in self.value.items():
            if isinstance(val, Node):
                out += str(key) + "=" + str(val.value)
            else:
                out += str(key) + "=" + str(val)
        return out

    def get_dot(self):
        """
        dot format of FunctionNode
        :return: string
        """
        out = '"' + self.key + '"' + "\t" + "->" + "\t"
        for key, val in self.value.items():
            if isinstance(val, Node):
                out += str(key) + "=" + str(val.value)
            else:
                out += str(key) + "=" + str(val)
        return out

    def recursive_dot(self, dictionary, count, name):
        """
        recursion function for dot format
        """
        super().recursive_dot(dictionary, count, name)
