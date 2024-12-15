import subprocess
from array import array

from .ast import (
    AST,
    ArrayDeclAST,
    FuncDeclAST,
    FuncDefnAST,
    IncludeAST,
    PrintfAST,
    ScanfAST,
)
from .node import FunctionNode, Node


class LLVM:
    def __init__(
        self, input_ast: AST | None = None, file_name: str = "../Output/Output.ll"
    ) -> None:
        """

        :param input_ast: the AST to be converted
        :param file_name: the file in which to write the LLVM code
        """
        super().__init__()
        self.ast = input_ast
        self.nodes = list()
        self.file_name = file_name
        self.index_queue = []

    def printf_int(self, value: int, index: int = 1):
        """
        Writes LLVM code for printf function printing an integer
        :param value: the value to print
        :param index: the current local index
        """
        out = ""
        out += "\t%" + str(index) + "= alloca i32, align 4 \n"
        out += "\tstore i32 " + str(value) + ", ptr %" + str(index) + ", align 4 \n"
        out += (
            "\t%" + str(index + 1) + "= load i32, ptr %" + str(index) + ", align 4 \n"
        )
        index += 1
        out += (
            "\t%"
            + str(index + 1)
            + "= call i32 (ptr, ...) @printf(ptr noundef @.str"
            + str(self.index_queue[0])
            + ", i32 noundef %"
            + str(index)
            + ")\n"
        )
        index += 2
        self.index_queue.remove(self.index_queue[0])
        return out, index

    def printf_float(self, val: float, index: int = 1):
        """
        Writes LLVM code for printf function printing a float
        :param val: the value to print
        :param index: the current local index
        """
        ll_out = ""
        # allocate the float parameter variable
        ll_out += "\t%" + str(index) + " = alloca float, align 4\n"
        # store the float value
        ll_out += "\tstore float " + str(val) + ", ptr %" + str(index) + ", align 4\n"
        index += 1
        ll_out += (
            "\t%" + str(index) + " = load float, ptr %" + str(index - 1) + ", align 4\n"
        )
        index += 1
        ll_out += (
            "\t%" + str(index) + " = fpext float %" + str(index - 1) + " to double\n"
        )
        index += 1
        ll_out += (
            "\t%"
            + str(index)
            + " = call i32 (ptr, ...) @printf(ptr noundef @.str"
            + str(self.index_queue[0])
            + ", double noundef %"
            + str(index - 1)
            + ")\n"
        )
        index += 1
        self.index_queue.remove(self.index_queue[0])
        return ll_out, index

    def printf_char(self, val: str, index: int = 1):
        """
        Writes LLVM code for printf function printing a character
        :param val: the value to print
        :param index: the current local index
        """
        ll_out = ""
        # allocate the float parameter variable
        ll_out += "\t%" + str(index) + " = alloca float, align 1\n"
        # store the float value
        ll_out += "\tstore i8 " + str(ord(val)) + ", ptr %" + str(index) + ", align 1\n"
        index += 1
        ll_out += (
            "\t%" + str(index) + " = load i8, ptr %" + str(index - 1) + ", align 1\n"
        )
        index += 1
        ll_out += "\t%" + str(index) + " = sext i8 %" + str(index - 1) + " to i32\n"
        index += 1
        ll_out += (
            "\t%"
            + str(index)
            + " = call i32 (ptr, ...) @printf(ptr noundef @.str"
            + str(self.index_queue[0])
            + ", i32 noundef %"
            + str(index - 1)
            + ")\n"
        )
        index += 1
        self.index_queue.remove(self.index_queue[0])
        return ll_out, index

    @staticmethod
    def sub(var_type: str, op1: str, op2: str):
        """
        Writes LLVM code for a subtract operation
        :param var_type: the type of return value
        :param op1: the first operand
        :param op2: the second operand
        """
        return f"sub nsw {var_type} {op1}, {op2}"

    @staticmethod
    def add(var_type: str, op1: str, op2: str):
        """
        Writes LLVM code for an addition operation
        :param var_type: the type of return value
        :param op1: the first operand
        :param op2: the second operand
        """
        return f"add nsw {var_type} {op1}, {op2}"

    @staticmethod
    def mul(var_type, op1, op2):
        """
        Writes LLVM code for a multiplication operation
        :param var_type: the type of return value
        :param op1: the first operand
        :param op2: the second operand
        """
        return f"mul nsw {var_type} {op1}, {op2}"

    @staticmethod
    def sdiv(var_type: str, op1: str, op2: str):
        """
        Writes LLVM code for a division operation (signed)
        :param var_type: the type of return value
        :param op1: the first operand
        :param op2: the second operand
        """
        return f"sdiv {var_type} {op1}, {op2}"

    @staticmethod
    def udiv(var_type: str, op1: str, op2: str):
        """
        Writes LLVM code for a division operation (unsigned)
        :param var_type: the type of return value
        :param op1: the first operand
        :param op2: the second operand
        """
        return f"udiv {var_type} {op1}, {op2}"

    @staticmethod
    def mod(var_type: str, op1: str, op2: str):
        """
        Writes LLVM code for a modulo operation (unsigned)
        :param var_type: the type of return value
        :param op1: the first operand
        :param op2: the second operand
        """
        return f"urem {var_type} {op1}, {op2}"

    @staticmethod
    def incr(var_type: str, op: str):
        """
        Writes LLVM code for an increment operation
        :param var_type: the type of return value
        :param op: the first operand
        """
        return LLVM.add(var_type, op, "1")

    @staticmethod
    def decr(var_type: str, op: str):
        """
        Writes LLVM code for a decrement operation
        :param var_type: the type of return value
        :param op: the first operand
        """
        return LLVM.sub(var_type, op, "1")

    @staticmethod
    def comp_gt(var_type: str, op1: str, op2: str):
        """
        Writes LLVM code for a greater than operation
        :param var_type: the type of return value
        :param op1: the first operand
        :param op2: the second operand
        """
        return f"icmp sgt {var_type} {op1}, {op2}"

    @staticmethod
    def comp_lt(var_type: str, op1: str, op2: str):
        """
        Writes LLVM code for a less than operation
        :param var_type: the type of return value
        :param op1: the first operand
        :param op2: the second operand
        """
        return f"icmp slt {var_type} {op1}, {op2}"

    @staticmethod
    def comp_eq(var_type: str, op1: str, op2: str):
        """
        Writes LLVM code for an is equal operation
        :param var_type: the type of return value
        :param op1: the first operand
        :param op2: the second operand
        """
        return f"icmp eq {var_type} {op1}, {op2}"

    @staticmethod
    def comp_geq(var_type: str, op1: str, op2: str):
        """
        Writes LLVM code for a greater than equals operation
        :param var_type: the type of return value
        :param op1: the first operand
        :param op2: the second operand
        """
        return f"icmp sge {var_type} {op1}, {op2}"

    @staticmethod
    def comp_leq(var_type: str, op1: str, op2: str):
        """
        Writes LLVM code for a less than equals operation
        :param var_type: the type of return value
        :param op1: the first operand
        :param op2: the second operand
        """
        return f"icmp sle {var_type} {op1}, {op2}"

    @staticmethod
    def comp_neq(var_type: str, op1: str, op2: str):
        """
        Writes LLVM code for a not equal operation
        :param var_type: the type of return value
        :param op1: the first operand
        :param op2: the second operand
        """
        return f"icmp ne {var_type} {op1}, {op2}"

    @staticmethod
    def and_op(var_type: str, op1: str, op2: str):
        """
        Writes LLVM code for an AND operation
        :param var_type: the type of return value
        :param op1: the first operand
        :param op2: the second operand
        """
        return f"and {var_type} {op1}, {op2}"

    @staticmethod
    def or_op(var_type: str, op1: str, op2: str):
        """
        Writes LLVM code for an OR operation
        :param var_type: the type of return value
        :param op1: the first operand
        :param op2: the second operand
        """
        return f"or {var_type} {op1}, {op2}"

    @staticmethod
    def not_op(var_type: str, op: str):
        """
        Writes LLVM code for an NOT operation
        :param var_type: the type of return value
        :param op: the first operand
        """
        return f"not {var_type} {op}"

    @staticmethod
    def assign(var_type: str, value, ptr):
        """
        Writes LLVM code for an assign operation
        :param var_type: the type of value to assign to the variable
        :param value: the value to store on the variable
        :param ptr: The register to store the value in
        """
        return f"store {var_type} {value}, {var_type}* {ptr}"

    def functionNodeConvert(
        self,
        func: FunctionNode,
        declr: bool = False,
        defn: bool = False,
        glob_decl: bool = False,
        index_local: int = 1,
        index_global: int = 1,
    ):
        """
        Writes LLVM code for function nodes according to the three parameters
        :param func: the function node
        :param declr: should function declaration be written
        :param defn: should function definition be written
        :param glob_decl: should the function declare global variables
        :param index_local: the current local index
        :param index_global: the current global index
        :return: (ll_out , index) || index
        ll_out: the LLVM code generated
        index: local index if in the tuple, global index if separate
        """
        # Output string, declared as empty
        ll_out = ""
        # Write the llvm code
        if declr:
            with open(self.file_name, "a") as f:
                if func.key == "printf":
                    f.write("declare i32 @printf(ptr noundef, ...)\n\n")
        if glob_decl:
            with open(self.file_name, "a") as f:
                # Check the function name
                if func.key == "printf":
                    # # Get the function parameters
                    # print_val = func.value["par0"]
                    # Define our string
                    std_decl = (
                        "@.str"
                        + str(index_global)
                        + " = private unnamed_addr constant ["
                    )
                    std_decl += "4"
                    std_decl += " x i8] c" + '"'
                    # if isinstance(print_val, VarNode):
                    #     if print_val.type == "int":
                    #         std_decl += "%d\\0A\\00\""
                    #     elif print_val.type == "float":
                    #         std_decl += "%f\\0A\\00\""
                    #     elif print_val.type == "char":
                    #         std_decl += "%c\\0A\\00\""
                    # else:
                    #     if print_val.key == "int":
                    #         std_decl += "%d\\0A\\00\""
                    #     elif print_val.key == "float":
                    #         std_decl += "%f\\0A\\00\""
                    #     elif print_val.type == "char":
                    #         std_decl += "%c\\0A\\00\""
                    if func.type == "int":
                        std_decl += '%d\\0A\\00"'
                    elif func.type == "float":
                        std_decl += '%f\\0A\\00"'
                    elif func.type == "char":
                        std_decl += '%c\\0A\\00"'
                    ll_out += std_decl + " align 1\n\n"
                f.write(ll_out)
                return index_global
        if defn:
            if func.body is not None:
                pass
        else:
            # Get function parameters
            print_val = func.value
            if isinstance(print_val, float):
                print_val = array("f", [print_val])[0]
                ret = self.printf_float(print_val, index_local)
                ll_out += ret[0]
                index_local = ret[1]
            elif isinstance(print_val, int):
                ret = self.printf_int(print_val, index_local)
                ll_out += ret[0]
                index_local = ret[1]
            elif isinstance(print_val, str):
                ret = self.printf_char(print_val, index_local)
                ll_out += ret[0]
                index_local = ret[1]
            return ll_out, index_local

    def ast_convert(self):
        """
        Writes LLVM code for an AST type object
        """
        list1 = [self.ast]
        check = True
        while check:
            for child in list1:
                if isinstance(child, PrintfAST):
                    continue
                if isinstance(child, AST):
                    index = list1.index(child)
                    list1.remove(child)
                    for _child in child.children:
                        list1.insert(index, _child)
            check = False
            for child in list1:
                if isinstance(child, PrintfAST):
                    continue
                if isinstance(child, AST):
                    check = True
        list1.reverse()
        # for child in list1:
        #     if isinstance(child, FunctionNode):
        #         self.functionNodeConvert(child)
        #     elif isinstance(child, VarNode):
        #         self.var_node_convert(child, False)  # True : Global False : Local
        self.nodes = list1

    def type_checker(self):
        """
        Not implemented yet
        """
        pass

    def convertNode(
        self, input_node: Node, global_scope: bool = False, index: int = 1
    ) -> int:
        """
        Writes LLVM code for a node type object
        """
        with open(self.file_name, "a") as f:
            # if isinstance(input_node , VarNode):
            #     f.write(self.var_node_convert(input_node , global_scope))
            if isinstance(input_node, PrintfAST):
                node = input_node.root
                node.value = input_node.children[0].value
                if len(self.index_queue) == 0:
                    self.index_queue.append(index)
                ret = self.functionNodeConvert(node, index_local=index)
                f.write(ret[0])
                index = ret[1]
            if isinstance(input_node, FunctionNode):
                if len(self.index_queue) == 0:
                    self.index_queue.append(index)
                ret = self.functionNodeConvert(input_node, index_local=index)
                f.write(ret[0])
                # self.index_queue.insert(0, ret[1])
                index = ret[1]
        return index

    def convert(self):
        """
        Converts the AST given in the constructor to LLVM code and writes the code to file
        """
        # clear file
        f = open(self.file_name, "w")
        # indexes = {"printf": 1}
        # get all the nodes via DFS
        # DFS the condition
        visited = list()
        not_visited = [self.ast]
        while len(not_visited) > 0:
            temp = not_visited.pop()
            if temp not in visited:
                # if a scope, skip
                # if include instruction, skip
                if (
                    isinstance(temp, FuncDeclAST)
                    or isinstance(temp, FuncDefnAST)
                    or isinstance(temp, PrintfAST)
                    or isinstance(temp, ScanfAST)
                    or isinstance(temp, ArrayDeclAST)
                    or isinstance(temp, IncludeAST)
                ):
                    visited.append(temp)
                if isinstance(temp, AST):
                    for child in temp.children:
                        not_visited.append(child)
        visited.reverse()
        string_global = ""
        string_local = ""
        index = 1
        for instruction in visited:
            if (
                isinstance(instruction, PrintfAST)
                or isinstance(instruction, ScanfAST)
                or isinstance(instruction, IncludeAST)
            ):
                temp_global, index = instruction.llvm_global(index)
            else:
                temp_global, index = instruction.llvm(False, index)
                # write the llvm code for all global variables for the instruction
                # temp_local, index = instruction.llvm(True, index)
                # string_local += temp_local
            string_global += temp_global
        f.write(string_global)
        f.write(string_local)
        print("Done!!!")
        # for entry in self.symbol_table.table:
        #     node = copy.copy(entry.object)
        #     if isinstance(entry, FuncSymbolEntry):
        #         node.key = entry.name
        #         if entry.name == "printf":
        #             i = indexes["printf"]
        #             defn = True
        #             declr = (i == 1)
        #             self.index_queue.append(i)
        #             i = self.functionNodeConvert(entry.object, declr=declr, defn=defn, glob_decl=True,
        #                                          index_global=i) + 1
        #             indexes["printf"] = i
        #     if isinstance(node, VarNode):
        #         self.var_node_convert(node, True)
        #     # elif isinstance(node, list):
        #     #     i = 1
        #     #     defn = True
        #     #     declr = True
        #     #     for val in node:
        #     #         if isinstance(entry, FunctionNode):
        #     #             self.index_queue.append(i)
        #     #             i = self.functionNodeConvert(entry, declr=declr, defn=defn, glob_decl=True, index_global=i) + 1
        #     #             defn = False
        #     #             declr = False
        # # begin of the main function
        # with open(self.file_name, 'a') as f:
        #     f.write("define dso_local i32 @main () {\n")
        #
        # self.ast_convert()
        # i = 1
        # for node in self.nodes:
        #     i = self.convertNode(node, False, index=i)
        #
        # # end of main
        # with open(self.file_name, 'a') as f:
        #     f.write("\tret i32 0\n}")

    def execute(self):
        """
        Runs the generated llvm file
        """
        subprocess.run(["lli", "-opaque-pointers", self.file_name])
