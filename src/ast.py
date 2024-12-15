import copy
import json
import os
import random
import re
import string
import uuid
from array import array
from math import floor
from typing import Any

import antlr4.error.ErrorListener
import antlr4.error.ErrorStrategy
from antlr4.error.Errors import ParseCancellationException

from .node import ArrayNode, FuncParameter, FunctionNode, Node, VarNode
from .register_management import Register, Registers
from .symbol_entry import FuncSymbolEntry, SymbolEntry
from .symbol_table import SymbolTable

# Standard Variables
keywords = [
    "var",
    "int",
    "binary_op",
    "unary_op",
    "comp_op",
    "comp_eq",
    "bin_log_op",
    "un_log_op",
    "assign_op",
    "const_var",
]
keywords_datatype = ["int", "float", "char"]

conversions = [
    ("float", "int"),
    ("int", "char"),
    ("float", "char"),
    ("int", "float"),
    ("char", "int"),
    ("char", "float"),
]
conv_promotions = [("int", "float"), ("char", "int"), ("char", "float")]
tokens = [
    "!=",
    "==",
    ">",
    ">=",
    "<",
    "<=",
    "||",
    "&&",
    "%",
    "/",
    "-",
    "+",
    "++",
    "--",
    "*",
]

# TODO: Replace code in the handle function of AstCreator with the handle functions


class ErrorListener(antlr4.error.ErrorListener.ErrorListener):
    def __init__(self):
        super().__init__()

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):  # noqa: N802, N803
        """
        Gives an error when there is a syntax error
        :return: a syntax error class
        """
        pass
        input_stream = (
            recognizer.getInputStream()
            if not hasattr(recognizer, "inputStream")
            else recognizer.inputStream
        )
        # Get all tokens in this line or the next one
        line_text = ""
        if hasattr(input_stream, "tokens"):
            for token in input_stream.tokens[input_stream.index :]:
                if token.line in range(line - 1, line + 1):
                    if input_stream.tokens.index(token) == input_stream.index:
                        line_text += "\u0332"
                    line_text += token.text
        else:
            line_text = input_stream.strdata.split("\n")[line - 1]
        out = f"Error at line {line!s}:{column!s} : {msg}\nLine where it occurred: {line_text}"
        raise ParseCancellationException(out)

    def reportAmbiguity(  # noqa: N802
        self,
        recognizer,
        dfa,
        startIndex,  # noqa: N803
        stopIndex,  # noqa: N803
        exact,
        ambigAlts,  # noqa: N803
        configs,
    ) -> None:
        """
        Gives an error when there is an ambiguity
        :return: None
        """
        # raise Exception("Ambiguity")
        super().reportAmbiguity(
            recognizer, dfa, startIndex, stopIndex, exact, ambigAlts, configs
        )


def isfloat(string) -> bool:
    """
    Checks if inout is a float
    :param string: input variable
    :return: bool
    """
    try:
        float(string)
        return True
    except ValueError:
        return False


def check_type(input: str):
    if input.isdigit():
        return "int"
    if isfloat(input):
        return "float"
    return "char"


def get_type(value):
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "char"
    return None


def get_type_from_format(value):
    if value == "d":
        return "int"
    if value == "f":
        return "float"
    if value == "c":
        return "char"
    if value == "s":
        return "string"
    return None


def convert(value, d_type):
    """
    help function for casting
    :param value: input_value
    :param d_type: cast type
    :return: cast value
    """
    try:
        if d_type == "int":
            if isinstance(value, int):
                return value
            if isinstance(value, str):
                return ord(value)
            return int(value)
        if d_type == "float":
            if isinstance(value, float):
                return value
            if isinstance(value, str):
                return float(ord(value))
            return float(value)
        if d_type == "char":
            if isinstance(value, str):
                return value
            return chr(value)
    except Exception as err:
        raise RuntimeError("Bad Cast") from err


def get_llvm_type(object_type: str) -> str | None:
    if object_type == "int":
        return "i32"
    if object_type == "float":
        return "float"
    if object_type == "char":
        return "i8"
    return None


def visited_list_dfs(ast) -> list:
    not_visited = [ast]
    visited = []
    while len(not_visited) > 0:
        v = not_visited.pop()
        if v not in visited:
            if not (isinstance(v, Node) or v is ast):
                visited.append(v)
            if isinstance(v, AST) and not isinstance(
                v, WhileLoopAST | FuncDeclAST | IfCondAST | FuncDefnAST | ForLoopAST
            ):
                for i in v.children:
                    if i is not ast:
                        not_visited.append(i)
    return visited


def visited_list_dfs2(ast) -> list:
    not_visited = [ast]
    visited = []
    while len(not_visited) > 0:
        v = not_visited.pop()
        if v not in visited:
            visited.append(v)
            if isinstance(v, Node):
                continue
            if isinstance(v, AST) and not isinstance(
                v, WhileLoopAST | FuncDeclAST | IfCondAST | FuncDefnAST | ForLoopAST
            ):
                for i in v.children:
                    if i is not ast:
                        not_visited.append(i)
    return visited


class AST:
    def __init__(
        self,
        root: Node | None = None,
        children: list | None = None,
        parent=None,
        symbol_table: SymbolTable | None = None,
    ):
        """
        Initializer function.
        :param root: Assign root node
        :param children: assign children if given
        """
        super().__init__()
        if children is None:
            children = []
        self.root: Node | None = root
        self.children: list[Node] | list[AST] | [] = children
        self.parent: AST | None = parent
        self.dic_count = {"instr": 0, "expr": 0}
        self.symbolTable: SymbolTable | None = symbol_table
        self.register = None
        self.blocks = []
        self.column = None
        self.line = None
        self.in_loop = False
        self.in_func = False
        self.stack_indexes = []

    def delete_unused_variables(self, filename: str | None = None):
        variables = []
        warnings = []
        for entry in self.symbolTable.table:
            if isinstance(entry, FuncSymbolEntry) and entry.symbol_table is not None:
                for entry2 in entry.symbol_table.table:
                    if not entry2.used:
                        variables.append(entry2)
                        # entry.symbol_table.remove(entry2)
            else:
                if not entry.used and entry.name not in ["printf", "scanf", "main"]:
                    variables.append(entry)
                    # self.symbolTable.remove(entry)
        variables.sort(
            key=lambda x: x.object.parent.line if x.object.parent is not None else 0,
            reverse=True,
        )
        visited = visited_list_dfs2(self)
        visited = [i for i in visited if isinstance(i, Node)]
        for entry in variables:
            entry.owner.remove(entry)
            temp_obj = entry.object
            # delete variable from its parent when:
            # it is only a declaration instruction
            # if temp_parent is not None and isinstance(temp_parent, InstrAST)
            # and len(temp_parent.children) == 1:
            #     temp_parent.children.remove(temp_obj)
            # fix the parents by deleting the instruction in which the variable is declared
            if temp_obj in visited:
                temp_obj = visited[visited.index(temp_obj)]
                temp_parent = temp_obj.parent
                while temp_parent is not None:
                    if (
                        isinstance(temp_parent, InstrAST)
                        and len(temp_parent.children) == 1
                    ):
                        temp_parent.parent.children.remove(temp_parent)
                    temp_parent = temp_parent.parent
            if filename:
                with open(filename) as f:
                    line = f.readlines()[entry.object.parent.line - 1]
                line = (
                    line[: entry.object.parent.column]
                    + "\u0332"
                    + line[entry.object.parent.column :]
                )
                warnings.append(
                    f"\033[95mwarning: \033[0m Unused variable {entry.name}\n"
                    f"{entry.object.parent.line}:{entry.object.parent.column}:\t{line}"
                )
        return warnings

    def variable_check(self):
        # DFS
        not_visited = [self]
        visited = []
        while len(not_visited) > 0:
            current = not_visited.pop()
            if current not in visited:
                visited.append(current)
                if isinstance(current, Node):
                    continue
                for i in current.children:
                    not_visited.append(i)

        visited.reverse()

        variables = []
        for i in visited:
            if isinstance(i, VarNode) or i.key == "var":
                variables.append(i)
        return variables

    def global_check(self) -> bool:
        """
        Check if the current node is in the global scope
        * yes: return True
        * no: return False
        :return: bool
        """
        if self.symbolTable is None:
            temp_parent = self.parent
            while temp_parent is not None:
                if isinstance(temp_parent, FuncScopeAST):
                    return False
                temp_parent = temp_parent.parent
            return True
        return self.symbolTable.parent is None

    def mips(self, registers: Registers):
        out_global = ""
        out_local = ""
        if self.parent is None:
            # declare symbol table in .data
            for entry in self.symbolTable.table:
                if isinstance(entry, FuncSymbolEntry):
                    continue
                if isinstance(entry.object, ArrayNode):
                    continue
                if entry.type == "int":
                    if entry.object.value in registers.globalObjects.data[2]:
                        continue
                    # out_global += f"{entry.name}: .word {entry.object.value if entry.object.value is not None else ''}\n"
                    registers.globalObjects.data[2][entry.object.value] = entry.name
                elif entry.type == "float":
                    if entry.object.value in registers.globalObjects.data[1]:
                        continue
                    # out_global += f"{entry.name}: .float {entry.object.value if entry.object.value is not None else ''}\n"
                    registers.globalObjects.data[1][entry.object.value] = entry.name
                elif entry.type == "char":
                    if entry.object.value in registers.globalObjects.data[0]:
                        continue
                    # out_global += f"{entry.name}: .byte {entry.object.value if entry.object.value is not None else ''}\n"
                    registers.globalObjects.data[0][entry.object.value] = entry.name
        # assign output registers to self
        registers.temporaryManager.LRU(self.root)
        self.register = self.root.register
        return out_local, out_global, []

    def search_blocks(self):
        if isinstance(self, WhileLoopAST):
            return self.blocks
        temp_parent = self.parent
        while self.parent is not None and isinstance(self.parent, ScopeAST):
            if isinstance(temp_parent, ScopeAST) and not isinstance(
                temp_parent, WhileLoopAST
            ):
                temp_parent = temp_parent.parent
            else:
                return temp_parent.blocks
        return None

    def llvm_global(self, index: int = 1) -> tuple[str, int]:
        """
        Generates the LLVM code for the global variables
        :param index: index of the current variable
        :return: tuple of the LLVM code and the index
        """
        out = ""
        return out, index

    def visit_llvm_op(self, current, index: int) -> tuple[str, int]:
        out = ""
        index_l = 0.0
        index_r = 0, 0
        left_child = current.children[0]
        right_child = None
        rentry = None
        lentry = None
        if len(current.children) > 1:
            right_child = current.children[1]

            if isinstance(right_child, VarNode):
                rentry, length = self.get_entry(right_child)
                if rentry is not None and rentry.register is None:
                    index_r = index
                    rentry.register = index
                    index += 1
                if rentry is not None:
                    index_r = rentry.register
                    out += f"\t%{index} load {get_llvm_type(right_child.type)}, ptr %{index_r}, align 4\n"
                    index += 1
            else:
                index_r = right_child.value

        if isinstance(left_child, VarNode):
            lentry, length = self.get_entry(left_child)
            if lentry is not None:
                if lentry is not None and lentry.register is None:
                    index_l = index
                    lentry.register = index
                    index += 1
                out += f"\t%{index} = load {get_llvm_type(left_child.type)}, ptr %{index_l}, align 4\n"
                index += 1
        else:
            index_l = left_child.value

        operand = current.root.value
        current_type = None
        if isinstance(left_child, VarNode):
            current_type = left_child.type
        elif isinstance(right_child, VarNode):
            current_type = right_child
        else:
            current_type = left_child.key
        converted_type = get_llvm_type(current_type)

        if operand == "<":
            out += f"\t%{index} = {self.comp_lt(converted_type, f"%{index_l}", f"%{index_r}")}\n"
        elif operand == ">":
            out += f"\t%{index} = {self.comp_gt(converted_type, f"%{index_l}", f"%{index_r}")}\n"
        elif operand == "==":
            out += f"\t%{index} = {self.comp_eq(converted_type, f"%{index_l}", f"%{index_r}")}\n"
        elif operand == "!=":
            out += f"\t%{index} = {self.comp_neq(converted_type, f"%{index_l}", f"%{index_r}")}\n"
        elif operand == "<=":
            out += f"\t%{index} = {self.comp_leq(converted_type, f"%{index_l}", f"%{index_r}")}\n"
        elif operand == ">=":
            out += f"\t%{index} = {self.comp_geq(converted_type, f"%{index_l}", f"%{index_r}")}\n"
        elif operand == "&&":
            out += f"\t%{index} = {self.and_op(converted_type, f"%{index_l}", f"%{index_r}")}\n"
        elif operand == "||":
            out += f"\t%{index} = {self.or_op(converted_type, f"%{index_l}", f"%{index_r}")}\n"
        elif operand == "+":
            out += f"\t%{index} = {self.add(converted_type, f"%{index_l}", f"%{index_r}")}\n"
        elif operand == "-":
            out += f"\t%{index} = {self.sub(converted_type, f"%{index_l}", f"%{index_r}")}\n"
        elif operand == "/":
            out += f"\t%{index} = {self.div(converted_type, f"%{index_l}", f"%{index_r}")}\n"
        elif operand == "*":
            out += f"\t%{index} = {self.mul(converted_type, f"%{index_l}", f"%{index_r}")}\n"
        elif operand == "%":
            out += f"\t%{index} = {self.mod(converted_type, f"%{index_l}", f"%{index_r}")}\n"
        elif operand == "++":
            out += f"\t%{index} = {self.incr(converted_type, f"%{index_l}")}\n"
        elif operand == "--":
            out += f"\t%{index} = {self.decr(converted_type, f"%{index_l}")}\n"

        if isinstance(current, CondAST):
            pass
        else:
            current.parent.children[current.parent.children.index(current)] = Node(
                current_type, index
            )
            index += 1
        return out, index

    def visit_mips_op(self, current, registers):
        out_local = ""
        out_global = ""
        out_list = []
        token = current.root.value
        left_register = None
        right_register = None
        left_type = None
        right_type = None
        left_node = copy.copy(current.children[0])
        right_node = None
        if len(current.children) > 1:
            right_node = copy.copy(current.children[1])
        # find the register for the variable
        left_register = registers.search(left_node)
        if right_node is not None:
            right_register = registers.search(right_node)
        # register assignment
        if isinstance(left_node, VarNode):
            if isinstance(left_node.parent, VarNode):
                registers.search(left_node.parent)
                left_register = left_node.parent.register.name
            elif left_register is None:
                registers.temporaryManager.LRU(left_node)
                left_register = left_node.register.name
            left_type = left_node.type
        else:
            if left_node.key == "var" and left_node.register is None:
                if left_register is None:
                    output = left_node.mips(registers)
                    out_local += output[0]
                    out_list += output[2]
                    left_register = left_node.register.name
            elif left_node.register is None:
                output = left_node.mips(registers)
                out_local += output[0]
                out_list += output[2]
                left_register = left_node.register.name
            else:
                left_register = left_node.register.name
        if right_node is not None:
            if isinstance(right_node, VarNode):
                right_register = right_node.register.name
                right_type = right_node.type
            elif isinstance(right_node, Node):
                if right_node.key == "var" and right_node.register is None:
                    if right_register is None:
                        output = right_node.mips(registers)
                        out_local += output[0]
                        out_list += output[2]
                        right_register = right_node.register.name
                elif right_node.register is None:
                    output = right_node.mips(registers)
                    out_local += output[0]
                    out_list += output[2]
                    right_register = right_node.register.name
                else:
                    right_register = right_node.register.name

        if left_type is None and left_register is not None:
            left_type = "float" if left_register.startswith("f") else "int"
        if right_type is None and right_register is not None:
            right_type = "float" if right_register.startswith("f") else "int"
        # create a new node
        new_node = Node("", None)
        if current.root.register is not None:
            new_node = current.root
        if new_node.register is None:
            if (right_type == "float" or left_type == "float") and token not in [
                "<",
                ">",
                "==",
                "!=",
                ">=",
                "<=",
            ]:
                registers.floatManager.LRU(new_node)
            else:
                registers.temporaryManager.LRU(new_node)
        new_register = new_node.register.name
        if new_register in [left_register, right_register]:
            # assign new register
            new_node = Node("", None)
            if new_node.register is None:
                if (right_type == "float" or left_type == "float") and token not in [
                    "<",
                    ">",
                    "==",
                    "!=",
                    ">=",
                    "<=",
                ]:
                    registers.floatManager.LRU(new_node)
                else:
                    registers.temporaryManager.LRU(new_node)
            new_register = new_node.register.name
        # casting if necessary
        if left_type != right_type and right_type is not None:
            if left_type == "float" and right_type == "int":
                registers.floatManager.LRU_delete(right_register)
                registers.floatManager.LRU(right_node)
                out_local += f"\tmtc1 ${right_register}, ${right_node.register.name}\n"
                right_register = right_node.register.name
                out_local += f"\tcvt.s.w ${right_register}, ${right_register}\n"
            elif right_type == "float" and left_type == "int":
                registers.floatManager.LRU_delete(left_register)
                registers.floatManager.LRU(left_node)
                out_local += f"\tmtc1 ${left_register}, ${left_node.register.name}\n"
                left_register = left_node.register.name
                out_local += f"\tcvt.s.w ${left_register}, ${left_register}\n"
            elif right_type == "float" and left_type == "char":
                # cast char to int first and then to float
                out_local += f"\tsubu ${left_register}, ${left_register}, 48\n"
                registers.floatManager.LRU_delete(left_register)
                registers.floatManager.LRU(left_node)
                out_local += f"\tmtc1 ${left_register}, ${left_node.register.name}\n"
                left_register = left_node.register.name
                out_local += f"\tcvt.s.w ${left_register}, ${left_register}\n"
            elif left_type == "float" and right_type == "char":
                # cast char to int first and then to float
                out_local += f"\tsubu ${right_register}, ${right_register}, 48\n"
                registers.floatManager.LRU_delete(right_register)
                registers.floatManager.LRU(right_node)
                out_local += f"\tmtc1 ${right_register}, ${right_node.register.name}\n"
                right_register = right_node.register.name
                out_local += f"\tcvt.s.w ${right_register}, ${right_register}\n"
            elif right_type == "int" and left_type == "char":
                registers.temporaryManager.LRU_delete(left_register)
                registers.temporaryManager.LRU(left_node)
                out_local += (
                    f"\tandi ${left_node.register.name}, ${left_register}, 0x000000FF\n"
                )
                left_register = left_node.register.name
            elif left_type == "int" and right_type == "char":
                registers.temporaryManager.LRU_delete(right_register)
                registers.temporaryManager.LRU(right_node)
                out_local += f"\tandi ${right_node.register.name}, ${right_register}, 0x000000FF\n"
                right_register = right_node.register.name
        out_list.append(left_register)
        out_list.append(right_register)
        out_list.append(new_register)
        if left_type == "float" or right_type == "float":
            temp_node = Node("", 1)
            temp2_node = Node("", None)
            if right_type is None:
                registers.floatManager.LRU(temp_node)
            else:
                registers.temporaryManager.LRU(temp_node)
            registers.temporaryManager.LRU(temp2_node)
            temp_save = temp_node.register.name
            temp2_save = (
                temp2_node.register.name if temp2_node.register is not None else None
            )
        uni = False
        if right_node is not None:
            # add commentaries
            if left_node.value is not None and left_node.key != "":
                out_local += f"\t\t\t# {left_node.key if isinstance(left_node, VarNode) else left_node.value}"
            else:
                out_local += f"\t\t\t# ${left_register}"
            out_local += f" {token} "
            if isinstance(right_node, Node):
                if right_node.value is not None and right_node.key != "":
                    out_local += f"{right_node.key if isinstance(right_node, VarNode) else right_node.value} --> ${new_register}\n"
                else:
                    out_local += f"${right_register} --> ${new_register}\n"
            if token == "<":
                if left_type == "float" or right_type == "float":
                    out_local += f"\tli ${temp_save}, 1\n"
                    out_local += f"\tc.le.s ${left_register}, ${right_register}\n"
                    out_local += f"\tmovt ${new_register}, ${temp_save}, 0\n"
                    out_local += f"\tmovf ${new_register}, $zero, 0\n"
                else:
                    out_local += (
                        f"\tslt ${new_register}, ${left_register}, ${right_register}\n"
                    )
            elif token == ">":
                if left_type == "float" or right_type == "float":
                    out_local += f"\tli ${temp2_save}, 0\n"
                    out_local += f"\tc.le.s ${left_register}, ${right_register}\n"
                    out_local += f"\tmovt ${new_register}, $zero, 0\n"
                    out_local += f"\tmovf ${new_register}, ${temp2_save}, 0\n"
                else:
                    out_local += (
                        f"\tslt ${new_register}, ${right_register}, ${left_register}\n"
                    )
            elif token == "==":
                if left_type == "float" or right_type == "float":
                    out_local += f"\tli ${temp_save}, 1\n"
                    out_local += f"\tc.eq.s ${left_register}, ${right_register}\n"
                    out_local += f"\tmovt ${new_register}, ${temp_save}, 0\n"
                    out_local += f"\tmovf ${new_register}, $zero, 0\n"
                else:
                    out_local += (
                        f"\tseq ${new_register}, ${left_register}, ${right_register}\n"
                    )
            elif token == "!=":
                if left_type == "float" or right_type == "float":
                    out_local += f"\tli ${temp2_save}, 0\n"
                    out_local += f"\tc.eq.s ${left_register}, ${right_register}\n"
                    out_local += f"\tmovt ${new_register}, $zero, 0\n"
                    out_local += f"\tmovf ${new_register}, ${temp2_save}, 0\n"
                else:
                    out_local += (
                        f"\tsne ${new_register}, ${left_register}, ${right_register}\n"
                    )
            elif token == "<=":
                if left_type == "float" or right_type == "float":
                    out_local += f"\tli ${temp_save}, 1\n"
                    out_local += f"\tc.le.s ${left_register}, ${right_register}\n"
                    out_local += f"\tmovt ${new_register}, ${temp_save}, 0\n"
                    out_local += f"\tmovf ${new_register}, $zero, 0\n"
                else:
                    out_local += (
                        f"\tsle ${new_register}, ${left_register}, ${right_register}\n"
                    )
            elif token == ">=":
                if left_type == "float" or right_type == "float":
                    out_local += f"\tli ${temp2_save}, 0\n"
                    out_local += f"\tc.lt.s ${left_register}, ${right_register}\n"
                    out_local += f"\tmovt ${new_register}, $zero, 0\n"
                    out_local += f"\tmovf ${new_register}, ${temp2_save}, 0\n"
                else:
                    out_local += (
                        f"\tsge ${new_register}, ${left_register}, ${right_register}\n"
                    )
            elif token == "&&":
                out_local += (
                    f"\tand ${new_register}, ${left_register}, ${right_register}\n"
                )
            elif token == "||":
                out_local += (
                    f"\tor ${new_register}, ${left_register}, ${right_register}\n"
                )
            elif token == "+":
                out_local += f"\tadd{'.s' if left_register == 'float' or right_register == 'float' else ''} ${new_register}, ${left_register}, ${right_register}\n"
            elif token == "-":
                out_local += f"\tsub{'.s' if left_register == 'float' or right_register == 'float' else ''} ${new_register}, ${left_register}, ${right_register}\n"
            elif token == "/":
                out_local += f"\tdiv{'.s' if left_register == 'float' or right_register == 'float' else ''} ${new_register}, ${left_register}, ${right_register}\n"
            elif token == "*":
                out_local += f"\tmul{'.s' if left_register == 'float' or right_register == 'float' else ''} ${new_register}, ${left_register}, ${right_register}\n"
            elif token == "%":
                out_local += (
                    f"\trem ${new_register}, ${left_register}, ${right_register}\n"
                )
        else:
            if token == "++":
                if left_type == "float":
                    out_local += f"\tli ${temp2_save}, 1\n"
                    out_local += f"\tmtc1 ${temp2_save}, ${temp_save}\n"
                    out_local += f"\tcvt.s.w ${temp_save}, ${temp_save}\n"
                    out_local += (
                        f"\tadd.s ${left_register}, ${left_register}, ${temp_save}\n"
                    )
                else:
                    out_local += f"\taddi ${left_register}, ${left_register}, 1\n"
                # out_local += f"\tmove ${left_register}, ${new_register}\n"
                uni = True
            elif token == "--":
                out_local += f"\taddi ${left_register}, ${left_register}, -1\n"
                # out_local += f"\tmove ${left_register}, ${new_register}\n"
                uni = True
            if isinstance(left_node, Node) and left_node.key == "var":
                if left_node.type == "int":
                    temp_type = "int"
                elif left_node.type == "float":
                    temp_type = "flt"
                else:
                    temp_type = "chr"
                out_local += f"\tsw{'c1' if left_node.type == 'float' else ''} ${left_register}, {temp_type}_{left_node.value}\n"
            elif isinstance(left_node, VarNode):
                if left_node.type == "int":
                    temp_type = "int"
                elif left_node.type == "float":
                    temp_type = "flt"
                else:
                    temp_type = "chr"
                if left_node.ptr:
                    pass
                    # if left_node.parent.register is None:
                    #     registers.temporaryManager.LRU(left_node.parent)
                    # store pointer value to address
                    # out_local += f"\tsw{'c1' if left_node.type == 'float' else ''} ${left_register}, {left_node.parent.type}_{left_node.parent.key}\n"
                else:
                    out_local += f"\tsw{'c1' if left_node.type == 'float' else ''} ${left_register}, {temp_type}_{left_node.key}\n"

        # shuffle used registers
        if left_register is not None:
            if left_register.startswith("t"):
                registers.temporaryManager.shuffle_name(left_register)
            else:
                registers.floatManager.shuffle_name(left_register)
        if right_register is not None:
            if right_register.startswith("t"):
                registers.temporaryManager.shuffle_name(right_register)
            else:
                registers.floatManager.shuffle_name(right_register)
        if new_register is not None:
            if new_register.startswith("t"):
                registers.temporaryManager.shuffle_name(new_register)
            else:
                registers.floatManager.shuffle_name(new_register)
        if uni:
            current.parent.children[current.parent.children.index(current)] = left_node
        else:
            current.parent.children[current.parent.children.index(current)] = new_node
        return out_local, out_global, [_ for _ in out_list if _ is not None]

    @staticmethod
    def get_entry(entry):
        if isinstance(entry, Node) and isinstance(entry.parent, ArrayNode):
            return entry.parent, 1
        out = None
        temp_symbol = None if isinstance(entry, Node) else entry.symbolTable
        temp_parent = entry.parent
        found = False
        while not found and temp_parent is not None:
            temp_symbol = (
                temp_parent.symbolTable if isinstance(temp_parent, AST) else None
            )
            temp_parent = temp_parent.parent
            if temp_symbol is not None:
                if temp_symbol.exists(entry):
                    out = temp_symbol.lookup(entry)
                    out[0].returned = True
                    return (out[0].object, len(out)) if out is not None else ([], None)
                if temp_symbol.exists(entry.value):
                    out = temp_symbol.lookup(entry.value)
                    out[0].returned = True
                    return (out[0].object, len(out)) if out is not None else ([], None)
        return out, -1

    def __repr__(self) -> str:
        return f"root: {{ {self.root} }} , children: {self.children}"

    def __sizeof__(self) -> int:
        return len(self.children)

    # def __getattr__(self, item):
    #     return

    def add_child(self, child):
        """
        Adds a child to the ast
        :param child: node
        :return: none
        """
        if child is None:
            return
        if not isinstance(child, AST) and not isinstance(child, Node):
            if not isinstance(child, AST):
                raise TypeError("child must be set to an AST")
            if not isinstance(child, Node):
                raise TypeError("child must be set to a Node")
        self.children.append(child)

    def save(self):
        """
        saves the ast in a dictionary
        :return: the dictionary
        """
        out, name = self.to_dict()
        if out[name] is None:
            out[name] = []
        else:
            out["children"] = []
        if self.root.value is None:
            out[name] = []
            for child in self.children:
                if isinstance(child, CommentAST):
                    continue
                out[name].append(child.save())
        else:
            out["children"] = [child.save() for child in self.children]
        return out

    def to_dict(self):
        return {self.root.key: self.root.value}, self.root.key

    def save_dot(self, dictionary_function: dict | None = None):
        """
        saves the ast in a dot format in a dictionary
        :return: dot format dictionary
        """
        if self.root.key in self.dic_count:
            name = f'"{self.root.key} {self.dic_count[self.root.key]}"'
            self.dic_count[self.root.key] += 1
        else:
            name = f'"{self.root.key}"'
        out = {name: self.root.value}
        if out[name] is None:
            out[name] = []
        else:
            out["children"] = []

        # The rest
        for i in range(len(self.children)):
            if (
                self.children[i] is not None
                and self.root.value is None
                and not isinstance(self.children[i], FunctionNode)
            ):
                out[name].append(self.children[i].save_dot())
            elif self.children[i] is not None and not isinstance(
                self.children[i], FunctionNode
            ):
                out["children"].insert(len(out["children"]), self.children[i].get_str())
        return out

    def print(self, indent: int = 4, save: bool = False, filename: str = ""):
        """
        prints a json format of ast
        :param filename: file to be saved into
        :param save: if True, it is saved to file
        :param indent: indent for json file
        :return:
        """
        output = self.save()
        if save:
            with open(f"../Output/{filename}.json", "w") as outfile:
                json.dump(output, outfile, indent=indent)
        print(json.dumps(output, indent=indent))

    def dot_language(self, file_name, symbol_table: dict | None = None):
        """
        Create dot language format file
        :param symbol_table:
        :param file_name: String that determines the file name
        :return: None
        """
        # Create file
        open("../Output/graphics/" + file_name + ".dot", "w+").close()
        # Start of dot language
        # self.recursive_dot(new_dictionary, count)
        self.connect("../Output/graphics/" + file_name + ".dot")
        # render the dot language file to png
        os.system(
            f"dot -Tpng ../Output/graphics/{file_name}.dot -o ../Output/graphics/{file_name}.png"
        )

    def connect(self, file_name: str):
        """
        connects the dictionary items together, to form a completed dot format file
        :return: None
        """
        with open(str(file_name), "w") as f:
            # A = AGraph(dictionary , directed=True)
            # A.graph_attr["shape"] = "tree"
            # A.write(file_name)
            # graph needs to be a spanning tree of nodes representing the ast
            out_begin = (
                'digraph G {\n\tgraph [ordering="out"];\n\tnode[shape=box, fontname="Liberation Sans"]\n\t'
                'edge[fontname="Liberation Sans", fontsize=10, penwidth=2, color="#000000"]\n\tlayout=dot;\n\t'
                'label="AST";\n\tsmoothing=avg_dist;\n\n'
            )
            f.write(out_begin)
            # for key, value in dictionary.items():
            #     string = ""
            #     for v in value:
            #         string += str(key) + "\t->\t"
            #         if isinstance(v, dict):
            #             for fk, fv in v.items():
            #                 string += fk + '\n'
            #                 string += f"subgraph {fk}" + '{\n'
            #                 mini_dict = {}
            #                 for i in range(len(fv)):
            #                     sub_string = ""
            #                     for j in fv[i]:
            #                         sub_string += j
            #                     string += '\t' + f"{fk};"
            #                     mini_dict[fk + str(i)] = sub_string
            #                 string += "}\n"
            #                 for mk, mv in mini_dict.items():
            #                     string += f"\"{mk[:-1]}\"" + "\t->\t" + f"\"{mv}\"" + "\n"
            #         else:
            #             string += "\t" + str(v) + "\n"
            #     f.write(string)

            # get all the nodes via DFS
            not_visited = copy.copy(self.children)
            visited = []
            while len(not_visited) > 0:
                current = not_visited.pop()
                visited.append(current)
                if isinstance(current, Node):
                    continue
                if isinstance(current, FuncScopeAST | ScopeAST):
                    current.children.reverse()
                for child in current.children:
                    if child is not None:
                        not_visited.append(child)
            visited.reverse()
            nodes = {}  # array to keep track of used names
            f.write("\t// Nodes:\n")
            # declare all the nodes. no edges yet
            for node in visited:
                if isinstance(node, FuncScopeAST | ScopeAST | IncludeAST):
                    continue
                new_key = node.key if isinstance(node, Node) else node.root.key
                if new_key not in nodes:
                    nodes[new_key] = [node]
                    if (
                        isinstance(node, InstrAST) and not isinstance(node, ReturnInstr)
                    ) or isinstance(node, FuncDefnAST):
                        continue
                    if isinstance(node, Node):
                        out = f'\t"{new_key}" [label="{node.value if node.value is not None else new_key}"];\n'
                    else:
                        out = f'\t"{new_key}" [label="{node.root.value if node.root.value is not None else new_key}"];\n'
                else:
                    nodes[new_key].append(node)
                    if isinstance(node, InstrAST | FuncDefnAST):
                        continue
                    if isinstance(node, Node):
                        out = f'\t"{new_key}_{len(nodes[new_key]) - 1!s}" [label="{node.value if node.value is not None else new_key}"];\n'
                    else:
                        out = f'\t"{new_key}_{len(nodes[new_key]) - 1!s}" [label="{node.root.value if node.root.value is not None else new_key}"];\n'

                f.write(out)
            f.write("\n\t// Edges:\n")
            in_func = False
            new_out = ""
            visited.reverse()
            # connect the nodes
            for node in visited:
                if isinstance(node, FuncScopeAST | ScopeAST):
                    if in_func:
                        f.write(new_out)
                        f.write("\t}\n")
                        in_func = False
                        new_out = ""
                    f.write(f"\tsubgraph cluster_{node.root.key} {{\n")
                    f.write(f'\t\tlabel="{node.root.key}"\n')
                    new_out = ""
                    in_func = True
                    continue
                if isinstance(node, Node | IncludeAST):
                    continue
                new_key = node.root.key
                if node.children is not None:
                    for child in node.children:
                        if isinstance(child, FuncScopeAST | ScopeAST):
                            continue
                        if isinstance(child, Node):
                            child_key = child.key
                        else:
                            child_key = child.root.key
                        # get the right node to connect to from the dictionary of nodes
                        index = None
                        child_index = None
                        for i in range(len(nodes[new_key])):
                            if isinstance(nodes[new_key][i], Node):
                                continue
                            if nodes[new_key][i] == child.parent:
                                # child = nodes[new_key][i]
                                index = i if i > 0 else None
                                break
                        for i in range(len(nodes[child_key])):
                            if nodes[child_key][i].parent == child.parent:
                                child_index = i if i > 0 else None
                                break
                        out = (
                            f"\t\"{new_key}{'_' + str(index) if index is not None else '' }\" -> "
                            f"\"{child.root.key if isinstance(child, AST) else child.key}"
                            f"{'_' + str(child_index) if child_index is not None else ''}\";\n"
                        )
                        new_out += out
                        if not in_func:
                            f.write(out)
            if in_func:
                new_out = new_out.replace("\t", "\t\t")
                f.write(new_out)
                f.write("\t}\n")
            f.write("}")

    def get_str(self):
        """
        string version of the root
        :return: str
        """
        return f"{self.root.key} : {self.root.value if self.root.value is not None else 'NaN'}"

    def get_dot(self):
        """
        dot format version of the root
        :return: str
        """
        return '"' + self.root.key + '"' + "\t" + "->" + "\t" + str(self.root.value)

    def handle(self):
        return self

    def llvm(self, scope: bool = False, index: int = 1) -> tuple[str, int]:
        out = ""
        # if len(self.children) > 0:
        #     for child in self.children:
        #         out , index = child.llvm(scope, index)
        return out, index

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
        return f"add nsw ptr {op1}, {op2}"

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
    def div(var_type: str, op1: str, op2: str):
        """
        Writes LLVM code for a division operation (unsigned)
        :param var_type: the type of return value
        :param op1: the first operand
        :param op2: the second operand
        """
        return AST.sdiv(var_type, op1, op2)

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
        return AST.add(var_type, op, "1")

    @staticmethod
    def decr(var_type: str, op: str):
        """
        Writes LLVM code for a decrement operation
        :param var_type: the type of return value
        :param op: the first operand
        """
        return AST.sub(var_type, op, "1")

    @staticmethod
    def comp_gt(var_type: str, op1: str, op2: str):
        """
        Writes LLVM code for a "greater than" operation
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
        Writes LLVM code for an "is equal" operation
        :param var_type: the type of return value
        :param op1: the first operand
        :param op2: the second operand
        """
        return f"icmp eq {var_type} {op1}, {op2}"

    @staticmethod
    def comp_geq(var_type: str, op1: str, op2: str):
        """
        Writes LLVM code for a "greater than equals" operation
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


class ExprAST(AST):
    def __init__(
        self, root: Node | None = None, children: list | None = None, parent=None
    ):
        super().__init__(root, children, parent)

    def handle(self):
        node = Node("", None, self.parent)
        for child in self.children:
            if isinstance(child, VarNode) and child.ptr:
                raise Exception("Cannot perform arithmetic operations on pointers")
        if isinstance(self.children[0], AST) or isinstance(self.children[1], AST):
            return self
        if self.children[0].key == "var" or self.children[1].key == "var":
            return self
        if self.children[0].value is None or self.children[1].value is None:
            return self

        if self.root.value == "+":
            node = self.children[0] + self.children[1]
            node_type = check_type(str(node.value))
            node.key = node_type
        elif self.root.value == "-":
            node = self.children[0] - self.children[1]
            node_type = check_type(str(node.value))
            node.key = node_type

        # convert the value of first operand to int
        if self.children[0].key == "char" and isinstance(self.children[0].value, str):
            temp_val1 = ord(self.children[0].value)
        else:
            temp_val1 = self.children[0].value

        # convert the value of second operand to int
        if self.children[1].key == "char" and isinstance(self.children[1].value, str):
            temp_val2 = ord(self.children[1].value)
        else:
            temp_val2 = self.children[1].value

        # perform the operation
        if self.root.value == "&&":
            node = Node("int", int(temp_val1 != 0 and temp_val2 != 0))
        elif self.root.value == "||":
            node = Node("int", int(temp_val1 != 0 or temp_val2 != 0))
        node.parent = self.parent
        return node

    def mips(self, registers: Registers):
        out_global = ""
        out_local = ""
        out_list = []
        if registers.search(self.root) is None:
            registers.temporaryManager.LRU(self.root)
        self.register = self.root.register
        # check if the expression is a constant
        if len(self.children) == 1 and isinstance(self.children[0], Node):
            out_local += f"li {self.register.name}, {self.children[0].value}\n"
            self.register.shuffle()
            return out_local, out_global, out_list
        # check if the operands have been assigned to registers
        if self.children[0].register is None:
            if self.children[0].key != "float":
                registers.temporaryManager.LRU(self.children[0])
            else:
                registers.floatManager.LRU(self.children[0])
        if self.children[1].register is None:
            if self.children[1].key != "float":
                registers.temporaryManager.LRU(self.children[1])
            else:
                registers.floatManager.LRU(self.children[1])
        temp_output = self.children[0].mips(registers)
        out_local += temp_output[0]
        out_global += temp_output[1]
        out_list += temp_output[2]
        temp_output = self.children[1].mips(registers)
        out_local += temp_output[0]
        out_global += temp_output[1]
        out_list += temp_output[2]
        output = self.visit_mips_op(self, registers)
        out_local += output[0]
        out_global += output[1]
        out_list += output[2]
        return out_local, out_global, out_list


class InstrAST(AST):
    def __init__(
        self, root: Node | None = None, children: list | None = None, parent=None
    ):
        super().__init__(root, children, parent)

    def handle(self):
        return self

    def mips(self, registers: Registers):
        out_global = ""
        out_local = ""
        out_list = []
        # if everything has been constant folded
        if len(self.children) == 1 and isinstance(self.children[0], Node):
            return self.children[0].mips(registers)
        for child in self.children:
            out = child.mips(registers)
            out_local += out[0]
            out_global += out[1]
            out_list += out[2]

        return out_local, out_global, out_list


class PrintfAST(AST):
    def __init__(
        self,
        root: Node | None = None,
        children: list | None = None,
        parent=None,
        format_string: str | None = None,
        args=None,
        format_specifiers: list | None = None,
    ):
        super().__init__(root, children, parent)
        if args is None:
            args = []
        self.format_string: str = format_string
        self.format_specifiers: list = []
        self.args: list = args
        self.width: int = 0

    def handle(self):
        warnings = []
        evaluate = True
        for child in self.children:
            child.parent = self
        # replace all the arguments
        for i in range(len(self.args)):
            if isinstance(self.args[i], Node) and self.args[i].key == "string":
                continue
            last_register = self.children[i].register
            self.args[i] = self.children[i]
            self.args[i].register = last_register
        # if any children haven't been replaced yet
        for i in range(len(self.args)):
            if isinstance(self.args[i], Node) and self.args[i].key == "var":
                evaluate = False
                # search for the variable in the symbol table
                temp_symbol = self.symbolTable
                while temp_symbol is None:
                    temp_symbol = self.parent.symbolTable
                while (
                    not temp_symbol.exists(self.args[i].value)
                    and temp_symbol.parent is not None
                ):
                    temp_symbol = temp_symbol.parent
                if temp_symbol.exists(self.args[i].value):
                    matches = temp_symbol.lookup(self.args[i].value)
                    if len(matches) > 1:
                        raise Exception("Ambiguous variable name")
                    self.args[i] = matches[0]
                    var_type = self.args[i].type
                    format_type = get_type_from_format(self.format_specifiers[i][-1])
                    if var_type == format_type:
                        continue
                    # check if the type of the variable matches
                    if (var_type, format_type) not in conversions and not self.args[
                        i
                    ].cast:
                        if (
                            var_type == "char"
                            and self.args[i].array
                            and format_type == "string"
                        ):
                            continue
                        raise Exception(
                            f"No possible conversion from {var_type} to {format_type}"
                        )
                    if (
                        var_type,
                        format_type,
                    ) not in conv_promotions and not self.args[i].cast:
                        # clang style warning
                        warnings.append(
                            f"Format specifies type '{format_type}' but the argument has type '{var_type}'\n"
                            f"Implicit conversion from '{var_type}' to '{format_type}'"
                        )
        if not evaluate:
            return self, warnings

        for i in range(len(self.format_specifiers)):
            current_specifier = self.format_specifiers[i]
            current_child = self.children[i]
            length = (
                int(current_specifier[1:-1]) if current_specifier[1:-1].isdigit() else 0
            )
            # check the child's type
            if current_specifier[-1] == "d":
                # check if the child is a node
                if isinstance(current_child, VarNode):
                    if current_child.type != "int":
                        if current_child.type == "float":
                            current_child.value = int(current_child.value)
                            if not current_child.cast:
                                warnings.append(
                                    "Implicit conversion from 'float' to 'int'"
                                )
                        elif current_child.type == "char":
                            current_child.value = ord(current_child.value)
                        elif current_child.value is None:
                            current_child.value = (
                                random.randint(0, 10**length)
                                if length > 0
                                else random.randint(0, 10)
                            )
                        else:
                            raise TypeError("Invalid type for printf")
                elif isinstance(current_child, Node) and not isinstance(
                    current_child.value, int
                ):
                    if isinstance(current_child.value, float):
                        current_child.value = int(current_child.value)
                        if not current_child.cast:
                            warnings.append("Implicit conversion from 'float' to 'int'")
                    elif (
                        isinstance(current_child.value, str)
                        and len(current_child.value) == 1
                    ):
                        current_child.value = ord(current_child.value)
                    elif current_child.value is None:
                        current_child.value = (
                            random.randint(0, 10**length)
                            if length > 0
                            else random.randint(0, 10)
                        )
                    else:
                        raise TypeError("Invalid type for printf")
                if isinstance(current_child, VarNode):
                    current_child.type = "int"
                else:
                    current_child.key = "int"
            if current_specifier[-1] == "i":
                # type can accept hexa, octal, decimal and binary
                if isinstance(current_child, VarNode):
                    if current_child.type != "int":
                        if current_child.type == "float":
                            current_child.value = int(current_child.value)
                            if not current_child.cast:
                                warnings.append(
                                    "Implicit conversion from 'float' to 'int'"
                                )
                        elif current_child.type == "char":
                            current_child.value = ord(current_child.value)
                        elif current_child.value is None:
                            current_child.value = random.randint(0, 10**length)
                        else:
                            raise TypeError("Invalid type for printf")
                elif isinstance(current_child, Node) and not isinstance(
                    current_child.value, int
                ):
                    if isinstance(current_child.value, float):
                        current_child.value = int(current_child.value)
                        if not current_child.cast:
                            warnings.append("Implicit conversion from 'float' to 'int'")
                    elif (
                        isinstance(current_child.value, str)
                        and len(current_child.value) == 1
                    ):
                        current_child.value = ord(current_child.value)
                    elif current_child.value is None:
                        current_child.value = random.randint(0, 10**length)
                    else:
                        raise TypeError("Invalid type for printf")

                if isinstance(current_child, VarNode):
                    current_child.type = "int"
                else:
                    current_child.key = "int"
            if current_specifier[-1] == "c":
                if isinstance(current_child, VarNode):
                    if current_child.type != "char":
                        if current_child.type == "float":
                            current_child.value = int(current_child.value)
                            if not current_child.cast:
                                warnings.append(
                                    "Implicit conversion from 'float' to 'char'"
                                )
                        elif current_child.type == "int":
                            current_child.value = chr(current_child.value)
                            if not current_child.cast:
                                warnings.append(
                                    "Implicit conversion from 'int' to 'char'"
                                )
                        elif current_child.value is None:
                            current_child.value = random.choice(string.ascii_letters)
                        else:
                            raise TypeError("Invalid type for printf")
                elif isinstance(current_child, Node):
                    if (
                        not isinstance(current_child.value, str)
                        or len(current_child.value) != 1
                    ):
                        if isinstance(current_child.value, float):
                            current_child.value = int(current_child.value)
                            if not current_child.cast:
                                warnings.append(
                                    "Implicit conversion from 'float' to 'char'"
                                )
                        elif isinstance(current_child.value, int):
                            current_child.value = chr(current_child.value)
                            if not current_child.cast:
                                warnings.append(
                                    "Implicit conversion from 'int' to 'char'"
                                )
                        elif current_child.value is None:
                            current_child.value = random.choice(string.ascii_letters)
                        else:
                            raise TypeError("Invalid type for printf")

                # if isinstance(current_child, VarNode):
                #     if current_child.type == "char":
                #         current_child.value = str(ord(current_child.value))
                else:
                    current_child.value = str(ord(current_child.value))
            if current_specifier[-1] == "f":
                if isinstance(current_child, VarNode):
                    if current_child.type != "float":
                        if current_child.type == "int":
                            current_child.value = array("f", [current_child.value])[0]
                        elif current_child.type == "char":
                            current_child.value = array(
                                "f", [ord(current_child.value)]
                            )[0]
                        elif current_child.value is None:
                            current_child.value = random.uniform(0, 10**length)
                        else:
                            raise TypeError("Invalid type for printf")
                elif isinstance(current_child, Node) and not isinstance(
                    current_child.value, float
                ):
                    if isinstance(current_child.value, int):
                        current_child.value = array("f", [current_child.value])[0]
                        warnings.append("Implicit conversion from 'int' to 'float'")
                    elif (
                        isinstance(current_child.value, str)
                        and len(current_child.value) == 1
                    ):
                        current_child.value = array("f", [ord(current_child.value)])[0]
                        if not current_child.cast:
                            warnings.append(
                                "Implicit conversion from 'char' to 'float'"
                            )
                    elif current_child.value is None:
                        current_child.value = random.uniform(0, 10**length)
                    else:
                        raise TypeError("Invalid type for printf")

            if current_specifier[-1] == "s":
                if isinstance(current_child, VarNode) and (
                    not current_child.type == "char" or not current_child.array
                ):
                    raise TypeError("Invalid type for printf")
                if isinstance(current_child, Node):
                    if current_child.value is None:
                        current_child.value = str(uuid.uuid1())
                    else:
                        if isinstance(current_child.value, str):
                            current_child.value = current_child.value
                        elif isinstance(current_child.value, int | float):
                            current_child.value = "\0"
                            warnings.append(
                                "Warning: format specifies type 'char *' but the argument has type 'int'"
                            )
                        else:
                            raise TypeError("Invalid type for printf")

            # check the length of format specifiers and child
            if (
                length != 0
                and isinstance(current_child, Node)
                and length > len(str(current_child.value))
            ):
                current_child.value = str(current_child.value).rjust(length, " ")
        return self, warnings

    def llvm_global(self, index: int = 1) -> tuple[str, int]:
        out = ""
        extra_length = len(re.findall("\\\.*", self.format_string)) * 2
        out += f'@.str.{index} = private unnamed_addr constant [{len(self.format_string)+1 - extra_length} x i8] c"{self.format_string}\\00", align 1\n'
        # entry, length = self.getEntry(self.root)
        self.register = index
        index += 1
        for i in range(len(self.args)):
            if self.args[i].key == "string":
                extra_length = len(re.findall("\\\.*", self.args[i].value)) * 2
                out += f'@str.{index} = private unnamed_addr constant [{len(self.args[i].value)+1 - extra_length} x i8] c"{self.args[i].value}\\00", align 1\n'
                self.args[i].register = index
                self.children[i].register = index
                index += 1
        return out, index

    def llvm(self, scope: bool = False, index: int = 1) -> tuple[str, int]:
        out = ""
        var_string = ""
        self.update_registers()
        for index, var in enumerate(self.args):
            temp_type = get_llvm_type(get_type(var.value))

            if isinstance(var, Node):
                if var.key == "string":
                    temp_type = "ptr"
                var_string += f"{temp_type if temp_type != 'float' else 'double'} noundef {'@str.' if var.key == 'string' else ''}{var.register if var.register is not None else var.value}"
            else:
                entry, length = self.get_entry(var)
                var += f"{temp_type if temp_type != 'float' else 'double'} noundef %{entry.register}"
            if index + 1 != len(self.args):
                var_string += ", "
        out += f"call i32 (ptr, ...) @printf(ptr noundef @.str.{self.register if self.register is not None else ''}{', ' if len(var_string) > 0 else ''}{var_string})\n"
        return out, index

    def update_registers(self):
        for i in range(len(self.args)):
            self.args[i].register = self.children[i].register

    def format(self, registers: Registers):
        # format string regex: ('%' ('-' | '+')? (INT)? [discf])*
        # we ignore the %s specifier because it is not used in the format string
        # keep everything in-between the format specifiers too
        format_ = re.split(r"(%[0-9]*[discf])|(\\[0-9A-Fa-f]{2})", self.format_string)
        format_ = [x for x in format_ if x is not None and x != ""]
        # loop through the list and check for valid format specifiers
        counter = -1
        for i in range(len(format_)):
            # if the string is a format specifier
            if format_[i].startswith("%"):
                counter += 1
                # remove the first character
                format_[i] = format_[i][1:]
                # if the first character is a number, it is the index of the argument to use
                # if the first character is a special character, it is a special format specifier
                # | Format Specifier | Output                                                       |
                # |------------------|--------------------------------------------------------------|
                # | `%10d`           | integer with a minimum width of 10 characters, right-aligned |
                # | `%-10s`          | string with a minimum width of 10 characters, left-aligned   |
                # | `%.2f`           | floating-point number with 2 decimal places of precision     |
                # | `%#x`            | hexadecimal integer with the `0x` prefix                     |
                # | `%+d`            | signed integer with a plus sign (+) for positive numbers     |
                # all valid specifiers
                left_align = False
                precision_valid = False
                precision = ""
                if format_[i][0] == "-":
                    left_align = True
                    format_[i] = format_[i][1:]
                elif format_[i][0] == ".":
                    # remove the dot
                    format_[i] = format_[i][1:]
                    # the next characters are the precision till the next character that is not a digit
                    for j in range(len(format_[i])):
                        if not format_[i][j].isdigit():
                            break
                        precision += format_[i][j]
                    # remove the precision from the format string
                    format_[i] = format_[i][len(precision) :]
                    # convert the precision to an integer
                    precision = int(precision)
                    precision_valid = True
                if format_[i][0].isdigit():
                    # take the full integer as the size
                    size_str = ""
                    for j in range(len(format_[i])):
                        if not format_[i][j].isdigit():
                            break
                        size_str += format_[i][j]
                    # remove the size from the format string
                    format_[i] = format_[i][len(size_str) :]
                    # convert the size to an integer
                    self.width = int(size_str)
                # replace the format specifier with the argument
                # check first if the format specifier is in the list of valid specifiers
                if format_[i][0] in [
                    "d",
                    "i",
                    "u",
                    "o",
                    "x",
                    "X",
                    "f",
                    "F",
                    "e",
                    "E",
                    "g",
                    "G",
                    "a",
                    "A",
                    "c",
                    "s",
                    "p",
                    "n",
                ]:
                    # if the format specifier is a string
                    temp_arg = self.args[counter]
                    arg = None
                    if isinstance(temp_arg, VarNode):
                        if isinstance(temp_arg, ArrayNode):
                            arg = temp_arg
                        elif temp_arg.value is not None:
                            arg = temp_arg.value
                        else:
                            registers.search(temp_arg)
                            arg = temp_arg
                    elif isinstance(temp_arg, Node):
                        if temp_arg.key == "var":
                            # search for the variable in the registers
                            registers.search(temp_arg)
                            arg = temp_arg
                            if temp_arg.register is not None:
                                arg = temp_arg.register
                            else:
                                arg = temp_arg.value
                        else:
                            arg = temp_arg.value
                    elif isinstance(temp_arg, SymbolEntry):
                        # search for the variable in the registers
                        registers.search(temp_arg.object)
                        arg = temp_arg.object
                    elif isinstance(temp_arg, FuncCallAST):
                        arg = temp_arg.root
                        arg = arg.register if arg.register is not None else arg.value
                    elif isinstance(temp_arg, ArrayElementAST):
                        arg = temp_arg
                        # if reg is not None:
                        #     arg = Register(in_name=f"{reg}")
                    # if the format specifier is an integer

                    elif isinstance(temp_arg, ArrayElementAST):
                        if temp_arg.register is None:
                            registers.search(temp_arg.root)
                            temp_arg.register = temp_arg.root.register
                        if isinstance(temp_arg.root, Node):
                            arg = temp_arg.root.register
                        elif isinstance(temp_arg.root, AST):
                            arg = temp_arg.root.value

                    if format_[i][0] in ["d", "i", "u", "o", "x", "X"]:
                        # if the precision is valid
                        if precision_valid:
                            format_[i] = arg
                            continue
                        # if the integer is shorter than the size
                        if isinstance(arg, str) and len(str(arg)) < self.width:
                            # if the integer is left-aligned
                            if left_align:
                                # add spaces to the right of the integer
                                arg = str(arg) + " " * (self.width - len(str(arg)))
                            # if the integer is right-aligned
                            else:
                                # add spaces to the left of the integer
                                arg = " " * (self.width - len(str(arg))) + str(arg)
                        format_[i] = arg
                    # if the format specifier is a floating-point number
                    elif format_[i][0] in ["f", "F", "e", "E", "g", "G", "a", "A"]:
                        # if the precision is valid
                        if isinstance(arg, str):
                            if precision_valid:
                                # if the number after the dot is longer than the precision
                                if len(str(arg).split(".")[1]) > precision:
                                    # truncate the number
                                    arg = str(arg)[:precision]
                                format_[i] = arg
                                continue
                            # if the number is shorter than the size
                            if len(str(arg)) < self.width:
                                # if the number is left-aligned
                                if left_align:
                                    # add spaces to the right of the number
                                    arg = str(arg) + " " * (self.width - len(str(arg)))
                                # if the number is right-aligned
                                else:
                                    # add spaces to the left of the number
                                    arg = " " * (self.width - len(str(arg))) + str(arg)
                        format_[i] = arg
                    # if the format specifier is a character
                    elif format_[i][0] == "c":
                        # if the precision is valid but is not float than continue
                        if isinstance(arg, str):
                            if precision_valid:
                                format_[i] = arg
                                continue
                            # if the character is shorter than the size
                            if len(str(arg)) < self.width:
                                # if the character is left-aligned
                                if left_align:
                                    # add spaces to the right of the character
                                    arg = str(arg) + " " * (self.width - len(str(arg)))
                                # if the character is right-aligned
                                else:
                                    # add spaces to the left of the character
                                    arg = " " * (self.width - len(str(arg))) + str(arg)
                        format_[i] = arg
                    # if the format specifier is a string
                    elif format_[i][0] == "s":
                        # if the precision is valid
                        if precision_valid:
                            format_[i] = arg
                            continue
                        # if the string is shorter than the size
                        if isinstance(arg, str) and len(str(arg)) < self.width:
                            # if the string is left-aligned
                            if left_align:
                                # add spaces to the right of the string
                                arg = str(arg) + " " * (self.width - len(str(arg)))
                            # if the string is right-aligned
                            else:
                                # add spaces to the left of the string
                                arg = " " * (self.width - len(str(arg))) + str(arg)
                        format_[i] = arg
        return format_

    def mips(self, registers: Registers):
        out_local = ""
        out_global = ""
        out_list = ["v0"]
        list_format = self.format(registers)
        registers.search(self.root)
        self.register = self.root.register
        # check all strings in list_format and if they are not in the global objects, add them
        for i in list_format:
            if isinstance(i, ArrayElementAST):
                continue
            if isinstance(i, VarNode):
                continue
            if isinstance(i, Register):
                continue
            if i in registers.globalObjects.data[0] or (
                (isinstance(i, str) and (len(i) == 0)) or isinstance(i, int)
            ):
                continue
            # if match regex r"\\[0-9A-Fa-f]{2}", then skip
            if isinstance(i, str) and re.match(r"\\[0-9A-Fa-f]{2}", i):
                continue
            if isinstance(i, float) and i not in registers.globalObjects.data[1]:
                # cast the float to be representable in mips
                if i not in registers.globalObjects.data[1]:
                    registers.globalObjects.data[1][i] = (
                        f"flt_{len(registers.globalObjects.data[1].items())}"
                    )
            elif isinstance(i, str):
                registers.globalObjects.data[0][i] = (
                    f"str_{len(registers.globalObjects.data[0].items())}"
                )
        # now syscall the list format in the right order with the right names
        for i in range(len(list_format)):
            # load the right variable in $a0
            # out_local += f"la $a0, {registers.globalObjects.data[0][i]}\n"
            # out_local += "li $v0, 4\n"
            # out_local += "syscall\n"
            # change so it calls the right print function
            if isinstance(list_format[i], str) and len(list_format[i]) == 0:
                continue
            if isinstance(list_format[i], ArrayElementAST):
                # resolve the array element ast
                node = list_format[i].root
                out_local += f"\tmov{'.s' if node.type == 'float' else 'e'} ${'f12' if node.type == 'float' else 'a0'}, ${node.register.name}\n"
                if node.type == "float":
                    out_local += "\tli $v0, 2\n"
                elif node.type == "int":
                    out_local += "\tli $v0, 1\n"
                elif node.type == "char":
                    out_local += "\tli $v0, 4\n"
                out_local += "\tsyscall\n"
            elif isinstance(list_format[i], ArrayNode):
                temp_node = Node("temp_node", None)
                registers.savedManager.LRU(temp_node)
                if registers.search(list_format[i]) is None:
                    if list_format[i].type == "float":
                        registers.floatManager.LRU(list_format[i])
                    else:
                        registers.temporaryManager.LRU(list_format[i])
                type_ = ""
                if list_format[i].type == "float":
                    type_ = "flt"
                elif list_format[i].type == "int":
                    type_ = "int"
                elif list_format[i].type == "char":
                    type_ = "chr"
                out_local += f"\tla ${list_format[i].register.name}, {type_}_{list_format[i].key}\n"
                out_local += f"\tli ${temp_node.register.name}, 0\n"
                temp_node.register.shuffle()
                out_local += f"\tadd ${list_format[i].register.name}, ${list_format[i].register.name}, ${temp_node.register.name}\n"
                out_local += "\tli $v0, 4\n"
                out_local += f"\tmov{'e' if list_format[i].type != 'float' else '.s'} $a0, ${list_format[i].register.name}\n"
                out_local += "\tsyscall\n"
                out_list.append("a0")
                out_list.append(temp_node.register.name)
                out_list.append(list_format[i].register.name)

            elif isinstance(list_format[i], str) and list_format[i].startswith("\\"):
                ascii_string = list_format[i].replace("\\", "")
                out_local += "\tli $v0, 11\n"
                out_local += f"\tla $a0, 0x{ascii_string}\n"
                out_local += "\tsyscall\n"
                if "a0" not in out_list:
                    out_list.append("a0")
                continue
            elif isinstance(list_format[i], Register):
                # get the register type
                if list_format[i].name[0] == "f":
                    out_local += f"\tmov.s $f12, ${list_format[i].name}\n"
                    out_local += "\tli $v0, 2\n"
                else:
                    out_local += f"\tmove $a0, ${list_format[i].name}\n"
                    if "a0" not in out_list:
                        out_list.append("a0")
                    # check a type of variable according to format
                    # keep everything in-between the format specifiers too
                    format_ = re.split(
                        r"(%[0-9]*[discf])|(\\[0-9A-Fa-f]{2})", self.format_string
                    )
                    format_ = [x for x in format_ if x is not None and x != ""]
                    if format_[i].endswith("d"):  # format for integer
                        out_local += "\tli $v0, 1\n"
                    elif format_[i].endswith("c"):  # format for float
                        out_local += "\tli $v0, 11\n"

                out_local += "\tsyscall\n"
            elif isinstance(list_format[i], int):  # if the type is an integer
                out_local += f"\tli $a0, {list_format[i]}\n"
                out_local += "\tli $v0, 1\n"
                out_local += "\tsyscall\n"
                if "a0" not in out_list:
                    out_list.append("a0")
            # temp fix for floats
            elif isinstance(list_format[i], float):  # if the type is a float
                registers.floatManager.LRU(self.root)
                self.register = self.root.register
                out_local += f"\tlwc1 ${self.register.name}, {registers.globalObjects.data[1][list_format[i]]}\n"
                out_local += f"\tmov.s $f12, ${self.register.name}\n"
                out_local += "\tli $v0, 2\n"
                out_local += "\tsyscall\n"
                if "f12" not in out_list:
                    out_list.append("f12")
            elif isinstance(list_format[i], str):  # if the type is a string/char
                out_local += (
                    f"\tla $a0, {registers.globalObjects.data[0][list_format[i]]}\n"
                )
                out_local += "\tli $v0, 4\n"
                out_local += "\tsyscall\n"
                if "a0" not in out_list:
                    out_list.append("a0")
            else:  # if the type is a variable
                var = list_format[i]
                out_local += "\tlw $a0, "
                if var.type == "int":
                    out_local += (
                        f"int_{var.key if isinstance(var, VarNode) else var.value}\n"
                    )
                    out_local += "\tli $v0, 1\n"
                elif var.type == "float":
                    out_local += (
                        f"flt_{var.key if isinstance(var, VarNode) else var.value}\n"
                    )
                    out_local += "\tli $v0, 2\n"
                elif var.type == "char":
                    out_local += (
                        f"chr_{var.key if isinstance(var, VarNode) else var.value}\n"
                    )
                    out_local += "\tli $v0, 4\n"
                out_local += "\tsyscall\n"
        return out_local, out_global, out_list


class DeclrAST(AST):
    def __init__(
        self,
        root: Node | None = None,
        children: list | None = None,
        parent=None,
        in_const: bool = False,
        var_type: str | None = None,
    ):
        super().__init__(root, children, parent)
        self.const = in_const
        self.type = var_type if var_type else ""

    def handle(self):
        return self

    def __repr__(self) -> str:
        return f"{super().__repr__()} {'const' if self.const else ''} {self.type} "

    def to_dict(self):
        return {
            self.root.key: [f"{'const ' if self.const else ''}{self.type}"]
        }, self.root.key

    def llvm(self, scope: bool = False, index: int = 1) -> tuple[str, int]:
        pass


class VarDeclrAST(AST):
    def __init__(
        self, root: Node | None = None, children: list | None = None, parent=None
    ):
        super().__init__(root, children, parent)

    def handle(self):
        if self.root and self.root.key == "assign":
            if not isinstance(self.children[0], VarNode):
                raise AttributeError("'Attempting to assign to a non-variable type'")
            # assign value
            if self.children[0].ptr:
                self.children[0].value = self.children[1]
                self.children[1].parent = self.children[0]
                # connect deref_level
                # self.children[0].total_deref += 1
                # self.children[1].total_deref = self.children[0].total_deref
                temp_value = self.children[0].value
                while isinstance(temp_value, VarNode):
                    temp_value.total_deref = self.children[0].total_deref
                    temp_value.deref_level += 1
                    temp_value = temp_value.value
            else:
                self.children[0].value = self.children[1].value
                self.children[0].cast = self.children[1].cast
            child = self.children[1].value
            while isinstance(child, VarNode):
                child = child.value
            self.children[0].type = get_type(child)
            # if isinstance(self.children[0], VarNode) and isinstance(self.children[1], VarNode):
            #     if self.children[0].total_deref != self.children[1].total_deref + 1:
            #         raise AttributeError(f"Incompatible types for {self.children[0].key} and {self.children[1].key}.")
            return self.children[0]
        return self.children[0]

    def llvm(self, scope: bool = False, index: int = 1) -> tuple[str, int]:
        pass


class AssignAST(AST):
    def __init__(
        self, root: Node | None = None, children: list | None = None, parent=None
    ):
        super().__init__(root, children, parent)

    def handle(self):
        # check if there are conversions needed
        return self.children[0]

    def llvm(self, scope: bool = False, index: int = 1) -> tuple[str, int]:
        out = ""
        return out, index

    def mips(self, registers: Registers):
        out_local = ""
        # check whether both operands have registers
        left_node = self.children[0]
        right_node = self.children[1]
        if left_node.register is None:
            if left_node.type == "float":
                registers.floatManager.LRU(left_node)
            else:
                registers.temporaryManager.LRU(left_node)
        if right_node.register is None:
            if right_node.key == "float":
                registers.floatManager.LRU(right_node)
            else:
                registers.temporaryManager.LRU(right_node)
        # assign value of right to left
        out_local += f"\tmove ${left_node.register.name}, ${right_node.register.name}\n"
        if left_node.type == "int":
            temp_type = "int"
        elif left_node.type == "float":
            temp_type = "flt"
        else:
            temp_type = "chr"
        if isinstance(left_node, Node):
            out_local += f"\tsw ${right_node.register.name}, {temp_type}_{left_node.value if left_node.key == 'var' else left_node.key}\n"
        elif isinstance(left_node, ArrayElementAST):
            out_local += (
                f"\tsw ${right_node.register.name}, 0(${left_node.register.name})\n"
            )
        return out_local, "", [right_node.register.name, left_node.register.name]


class TermAST(AST):
    def __init__(
        self, root: Node | None = None, children: list | None = None, parent=None
    ):
        super().__init__(root, children, parent)

    def handle(self):
        node = Node("", None)
        warnings = []
        for child in self.children:
            if isinstance(child, AST):
                return self
            if isinstance(child, VarNode) and child.ptr:
                raise RuntimeError("'Cannot use pointer as a term'")
            if child.value is None and not isinstance(child, ArrayNode):
                return self
            if child.key == "var":
                return self
        if self.root.value == "*":
            node = self.children[0] * self.children[1]
            node_type = check_type(str(node.value))
            node.key = node_type
        elif self.root.value == "%":
            node = self.children[0] % self.children[1]
            node_type = check_type(str(node.value))
            node.key = node_type
        elif self.root.value == "/":
            node = self.children[0] / self.children[1]
            if self.children[0].key != "float" and self.children[1].key != "float":
                node.value = floor(node.value)
        elif self.root.value == "++":
            if len(self.children) != 1:
                raise RuntimeError(
                    "'Expected one variable for increment operation, got multiple'"
                )
            if (
                not isinstance(self.children[0], VarNode)
                and not self.children[0].parent.array
            ):
                raise AttributeError(
                    "'Attempting to decrement a non-variable type object'"
                )
            node = self.children[0]
            if isinstance(node, VarNode):
                if node.const:
                    raise AttributeError(
                        f"'Attempting to modify a const variable {node.key}'"
                    )
            else:
                if node.parent.const:
                    raise AttributeError(
                        f"'Attempting to modify a const variable {node.key}'"
                    )
            node.value += 1
        elif self.root.value == "--":
            if len(self.children) != 1:
                raise RuntimeError(
                    "'Expected one variable for increment operation, got multiple'"
                )
            if (
                not isinstance(self.children[0], VarNode)
                and not self.children[0].parent.array
            ):
                raise AttributeError(
                    "'Attempting to decrement a non-variable type object'"
                )
            node = self.children[0]
            if isinstance(node, VarNode):
                if node.const:
                    raise AttributeError(
                        f"'Attempting to modify a const variable {node.key}'"
                    )
            else:
                if node.parent.const:
                    raise AttributeError(
                        f"'Attempting to modify a const variable {node.key}'"
                    )
            node.value -= 1
            # node.value = 0
            # node.key = "int"
        if self.root.value == "<=":
            node = self.children[0] <= self.children[1]
            node.value = int(node.value)
        elif self.root.value == "<":
            node = self.children[0] < self.children[1]
            node.value = int(node.value)
        elif self.root.value == ">=":
            node = self.children[0] >= self.children[1]
            node.value = int(node.value)
        elif self.root.value == ">":
            node = self.children[0] > self.children[1]
            node.value = int(node.value)
        elif self.root.value == "==":
            if not isinstance(self.children[0], type(self.children[1])):
                node.value = self.children[0].value == self.children[1].value
            else:
                node.value = self.children[0] == self.children[1]
            node.value = int(node.value)
            node.key = "int"
        elif self.root.value == "!=":
            if not isinstance(self.children[0], type(self.children[1])):
                node.value = self.children[0].value != self.children[1].value
            else:
                node.value = self.children[0] != self.children[1]
            node.value = int(node.value)
            node.key = "int"
        elif self.root.value == "!":
            if self.children[0].key == "char":
                node.value = not ord(self.children[0].value)
            node.value = not self.children[0].value
            node.value = int(node.value)
            node.key = "int"
        node.parent = self.parent
        if self.root.value in ["==", "!=", "<", "<=", ">", ">="]:
            if (
                isinstance(self.children[0], ArrayNode)
                and not isinstance(self.children[1], ArrayNode)
            ) or (
                not isinstance(self.children[0], ArrayNode)
                and isinstance(self.children[1], ArrayNode)
            ):
                raise RuntimeError("'Attempting to compare array with non-array'")
            if isinstance(self.children[0], ArrayNode) and isinstance(
                self.children[1], ArrayNode
            ):
                # add clang style warning with pink color
                warnings.append("array comparison always evaluates to false\n")
                return node, warnings
        return node

    def mips(self, registers: Registers):
        out = ""
        out_global = ""
        out_list = []
        out, out_global, out_list = self.visit_mips_op(self, registers)

        return out, out_global, out_list


class FactorAST(AST):
    def __init__(
        self, root: Node | None = None, children: list | None = None, parent=None
    ):
        super().__init__(root, children, parent)

    def handle(self):
        if self.children[0].key == "var":
            return self
        if self.children[0].value is None:
            return self
        if isinstance(self.children[0], VarNode) and self.children[0].ptr:
            if self.root.value == "++":
                raise RuntimeError("'Attempting to increment a pointer'")
            if self.root.value == "--":
                raise RuntimeError("'Attempting to decrement a pointer'")
            if self.root.value == "-":
                raise RuntimeError("'Attempting to negate a pointer'")
            if self.root.value == "+":
                raise RuntimeError("'Invalid operation on pointer'")
        if self.root.value == "-":
            self.children[0].value = -self.children[0].value
            return self.children[0]
        if self.root.value == "+":
            return self.children[0]
        if self.root.value == "++":
            if len(self.children) != 1:
                raise RuntimeError(
                    "'Expected one variable for increment operation, got multiple'"
                )
            if (
                not isinstance(self.children[0], VarNode)
                and not self.children[0].parent.array
            ):
                raise AttributeError(
                    "'Attempting to decrement a non-variable type object'"
                )
            node = self.children[0]
            if isinstance(node, VarNode):
                if node.const:
                    raise AttributeError(
                        f"'Attempting to modify a const variable {node.key}'"
                    )
            else:
                if node.parent.const:
                    raise AttributeError(
                        f"'Attempting to modify a const variable {node.key}'"
                    )
            node.value += 1
            return node
        if self.root.value == "--":
            if len(self.children) != 1:
                raise RuntimeError(
                    "'Expected one variable for increment operation, got multiple'"
                )
            if (
                not isinstance(self.children[0], VarNode)
                and not self.children[0].parent.array
            ):
                raise AttributeError(
                    "'Attempting to decrement a non-variable type object'"
                )
            node = self.children[0]
            if isinstance(node, VarNode):
                if node.const:
                    raise AttributeError(
                        f"'Attempting to modify a const variable {node.key}'"
                    )
            else:
                if node.parent.const:
                    raise AttributeError(
                        f"'Attempting to modify a const variable {node.key}'"
                    )
            node.value -= 1
            return node
        return None


class PrimaryAST(AST):
    def __init__(
        self, root: Node | None = None, children: list | None = None, parent=None
    ):
        super().__init__(root, children, parent)

    def handle(self):
        if self.root.value == "&":
            if isinstance(self.children[0], Node):
                self.children[0].addr = True
            return self.children[0]
        if self.root.value[0] + self.root.value[-1] == "()":
            ret = self.children[0]
            cast = self.root.value[1:-1]
            ret.value = convert(ret.value, cast)
            ret.cast = True
            return self.children[0]
        return self


class DerefAST(AST):
    def __init__(
        self, root: Node | None = None, children: list | None = None, parent=None
    ):
        super().__init__(root, children, parent)

    def handle(self):
        child = self.children[0]
        if isinstance(child, Node) and child.key == "var":
            temp_parent = child.parent
            while temp_parent is not None:
                if temp_parent.symbolTable is not None:
                    break
                temp_parent = temp_parent.parent
            temp_symbol = temp_parent.symbolTable
            while not temp_symbol.exists(child.value):
                temp_symbol = temp_symbol.parent
            match = temp_symbol.lookup(child.value)
            match = Node("var", match[0].object.value.key)
            match.parent = self.parent
            return match
        if not isinstance(child, VarNode):
            raise ReferenceError("Attempting to dereference a non-variable type object")
        if not child.ptr:
            raise AttributeError(
                "Attempting to dereference a non-pointer type variable"
            )
        if child.deref_level > child.total_deref:
            raise AttributeError(f"Dereference depth reached for pointer {child.key}")

        if child.value is None:
            # check the type of the pointer
            if child.type == "int":
                out = Node("int", 0)
                out.parent = self.children[0]
            elif child.type == "float":
                out = Node("float", 0.0)
                out.parent = self.children[0]
            elif child.type == "char":
                out = Node("char", "")
                out.parent = self.children[0]
            return out
        child = child.value
        child.deref = True
        child.known = self.children[0].known
        if not child.known:
            new_node = Node("var", child.key)
            new_node.parent = self.parent
            child = new_node
        return child

    def mips(self, registers: Registers):
        out_local = ""
        out_global = ""
        out_list = []
        if self.children[0].value is not None:
            out_local, out_global, out_list = self.children[0].value.mips(registers)
        return out_local, out_global, out_list


class ArrayElementAST(AST):
    def __init__(
        self, root: Node | None = None, children: list | None = None, parent=None
    ):
        super().__init__(root, children, parent)
        self.array: ArrayNode | None = None
        self.type: str | None = None

    def handle(self):
        # update root value
        if len(self.children) == 2 and isinstance(self.children[1], Node):
            self.root.value = self.children[1].value
        # get nearest symbol table
        temp_symbol = self.symbolTable
        temp_parent = self.parent
        var_index = None
        if isinstance(self.root.value, str) and var_index is None:
            var_index = self.root.value
        while temp_symbol is None and temp_parent is not None:
            temp_symbol = temp_parent.symbolTable
            temp_parent = temp_parent.parent
        if temp_symbol is None:
            raise AttributeError(f"Array {self.root.key} not found in symbol table")
        while not temp_symbol.exists(self.root.key):
            if temp_symbol.exists(var_index):
                self.root.value = temp_symbol.lookup(var_index)[0].object
            temp_symbol = temp_symbol.parent
            if temp_symbol is None:
                raise AttributeError(f"Array {self.root.key} not found in symbol table")
        matches = temp_symbol.lookup(self.root.key)
        temp_symbol = temp_parent.symbolTable
        if var_index == self.root.value:
            # search for variable in symbol table too
            while not temp_symbol.exists(var_index):
                temp_symbol = temp_symbol.parent
                if temp_symbol is None:
                    raise AttributeError(
                        f"Variable {var_index} not found in symbol table"
                    )
        if var_index is not None:
            self.root.value = temp_symbol.lookup(var_index)[0].object
        if len(matches) != 1:
            raise AttributeError(f"Multiple definitions of array {self.root.key}")
        temp_symbol = matches[0]
        self.type = temp_symbol.type
        matches[0].used = True
        if isinstance(self.root.value, str | Node):
            return self
        if not isinstance(self.root.value, int):
            raise AttributeError("Array index must be an integer")
        if not isinstance(temp_symbol.object, ArrayNode):
            raise AttributeError("Attempting to index a non-array type object")
        if self.root.value < 0 or self.root.value >= temp_symbol.size:
            raise AttributeError(
                f"Array index {self.root.value} out of bounds for array {self.root.key}"
            )
        return temp_symbol.object.values[self.root.value]

    def save(self):
        # for printing purposes
        return f"{self.root.key}[{self.root.value}]"

    def mips(self, registers: Registers):
        out_local = out_global = ""
        out_list = []
        # steps to load an array element with register in a variable
        # 1. load array address in a register - done before calling this function
        # 2. multiply the index with the size of the array element
        temp_node = Node("*", None)
        registers.temporaryManager.LRU(temp_node)
        if self.children[1].register is None:
            registers.temporaryManager.LRU(self.children[1])
        if self.register is None:
            registers.temporaryManager.LRU(self.root)
            self.register = self.root.register
        # sll register_of_offset, index, 2
        if self.children[0].type == "int" or self.children[0].type == "float":
            out_local += f"\tsll ${temp_node.register.name}, ${self.children[1].register.name}, 2\n"
        else:
            out_local += f"\tmove ${temp_node.register.name}, ${self.children[1].register.name}\n"
        # 3. add the offset to the array address
        # out_local += f"\tadd ${self.children[0].register.name}, ${self.children[0].register.name}, ${temp_node.register.name}\n"
        # 4. load the value at the address in a register
        type_ = ""
        if self.children[0].type == "int":
            type_ = "int"
        elif self.children[0].type == "float":
            type_ = "flt"
        elif self.children[0].type == "char":
            type_ = "chr"
        if self.children[0].type == "char":
            out_local += f"\tlb ${self.register.name}, {type_}_{self.children[0].key}(${temp_node.register.name})\n"
        else:
            out_local += f"\tlb ${self.register.name}, {type_}_{self.children[0].key}(${temp_node.register.name})\n"
        out_list.append(temp_node.register.name)
        out_list.append(self.children[1].register.name)
        out_list.append(self.register)
        registers.shuffle_name(self.register.name)
        return out_local, out_global, out_list


class ScopeAST(AST):
    def __init__(
        self,
        root: Node | None = None,
        children: list | None = None,
        parent=None,
        condition: AST | None = None,
    ):
        super().__init__(root, children, parent, symbol_table=SymbolTable())
        self.condition: AST | Node | None = condition
        self.symbolTable.owner = self

    def handle(self):
        return self

    def llvm(self, scope: bool = False, index: int = 1) -> tuple[str, int]:
        visited = visited_list_dfs(self)
        out = ""

        for current in visited:
            output = tuple
            if current.root.value in tokens:
                output = self.visit_llvm_op(current, index)
            else:
                output = current.llvm(True, index)
            out += output[0]
            index = output[1]
        return out, index

    def mips(self, registers: Registers):
        visited = []
        not_visited = self.children
        # not_visited.reverse()
        # DFS
        while len(not_visited) != 0:
            current = not_visited.pop()
            if current not in visited:
                visited.append(current)
                if isinstance(current, AST) and not (
                    isinstance(
                        current,
                        ScopeAST
                        | FuncDefnAST
                        | FuncCallAST
                        | IfCondAST
                        | WhileLoopAST
                        | ForLoopAST
                        | FuncDeclAST
                        | SwitchAST,
                    )
                ):
                    for i in current.children:
                        if isinstance(i, ArrayDeclAST) and isinstance(
                            current, InstrAST
                        ):
                            continue
                        not_visited.append(i)
        visited.reverse()
        out_local = out_global = ""
        out_list = []

        output = None
        original_stacksize = registers.globalObjects.stackSize
        for current in visited:
            if isinstance(current, Node):
                output = current.mips(registers)
            elif current.root.value in tokens:
                output = self.visit_mips_op(current, registers)
            else:
                output = current.mips(registers)
            out_local += output[0]
            out_global += output[1]
            out_list += output[2]
        if original_stacksize != registers.globalObjects.stackSize:
            pass
        # # begin
        return out_local, out_global, out_list


class IfCondAST(ScopeAST):
    def __init__(
        self, root: Node | None = None, children: list | None = None, parent=None
    ):
        super().__init__(root, children, parent)
        self.exit: int = -1

    def handle(self):
        return self

    def to_dict(self):
        return {"if": self.condition.save()}, "if"

    def save(self):
        out, name = self.to_dict()
        if out[name] is None:
            out[name] = []
        else:
            out["body"] = []
        if self.condition is None:
            out[name] = [child.save() for child in self.children]
        else:
            out["body"] = [child.save() for child in self.children]
        return out

    def llvm(self, scope: bool = False, index: int = 1) -> tuple[str, int]:
        # condition first with branch
        out = ""
        visited = visited_list_dfs(self.condition)

        for current in visited:
            output_str, output_in = self.visit_llvm_op(current, index)
            out += output_str
            index = output_in

        # if block
        visited = visited_list_dfs(self.children[0])

        for current in visited:
            output = tuple
            if current.root.value in tokens:
                output = self.visit_llvm_op(current, index)
            else:
                output = current.llvm(True, index)
            out += output[0]
            index = output[1]

        # else block if else ast exist do 'else llvm' else do 'create block'
        else_bool = False
        for child in self.children:
            if isinstance(child, ElseCondAST):
                output = child.llvm(True, index)
                out += output[0]
                index = output[1]
                else_bool = True
                break
        # create new block if else llvm didn't pass
        if not else_bool:
            out += f"{index}:\n"
            index += 1

        return out, index

    def mips(self, registers: Registers):
        # condition first
        # DFS
        temp_list = out_list = []
        out_global = out_cond = ""
        # condition check and branch
        out_cond, temp_glob, temp_list = self.condition.mips(registers)
        out_list += temp_list
        out_local = f"if_{registers.globalObjects.index}: \n"
        self.register = registers.globalObjects.index
        registers.globalObjects.index += 1
        return_bool = False
        if len(self.children[0].children) > 0 and isinstance(
            self.children[0].children[0], ReturnInstr
        ):
            return_bool = True
        # body of the if loop
        output = self.children[0].mips(registers)
        temp_out = output[0]
        temp_list += output[2]
        temp_list.append("v1")
        # condition
        out_local += out_cond
        if isinstance(self.children[-1], ElseCondAST):
            # branch if condition is false to else block
            out_local += f"\tbeq $v1, $zero, else_{registers.globalObjects.index}\n"
            out_else = f"else_{registers.globalObjects.index}: \n"
        else:
            out_local += f"\tbeq $v1, $zero, exit_{registers.globalObjects.index}\n"
            self.exit = registers.globalObjects.index
            out_else = ""
        # if "else" block exist then create else default
        if len(self.children) > 1:
            self.children[1].register = registers.globalObjects.index
            output = self.children[1].mips(registers)
            out_else += output[0]
            out_list += output[2]
            out_global += output[1]
        else:
            pass
        registers.globalObjects.index += 1
        out_local += temp_out
        if self.exit == -1:
            self.exit = registers.globalObjects.index
        if not return_bool:
            out_local += f"\tj exit_{self.exit}\n"
        out_local += out_else
        if not return_bool:
            out_local += f"exit_{self.exit}: \n"
        registers.globalObjects.index += 1
        out_list += temp_list
        return out_local, out_global, out_list


class ElseCondAST(ScopeAST):
    def __init__(
        self, root: Node | None = None, children: list | None = None, parent=None
    ):
        super().__init__(root, children, parent)

    def to_dict(self):
        return {"else": None}, "else"

    def save(self):
        out, name = self.to_dict()
        if out[name] is None:
            out[name] = [child.save() for child in self.children]
        return out

    def handle(self):
        return self

    def llvm(self, scope: bool = False, index: int = 1) -> tuple[str, int]:
        visited = visited_list_dfs(self.children[0])
        out = ""
        for current in visited:
            output = tuple
            if current.root.value in tokens:
                output = self.visit_llvm_op(current, index)
            else:
                output = current.llvm(True, index)
            out += output[0]
            index = output[1]
        return out, index

    def mips(self, registers: Registers):
        return self.children[0].mips(registers)


class ForLoopAST(ScopeAST):
    def __init__(
        self, root: Node | None = None, children: list | None = None, parent=None
    ):
        super().__init__(root, children, parent)
        self.initialization = None
        self.incr = None

    def handle(self):
        return self

    def to_dict(self):
        return {
            "for": [self.initialization.save(), self.condition.save(), self.incr.save()]
        }, "for"

    def save(self):
        out, name = self.to_dict()
        if out[name] is None:
            out[name] = []
        if self.condition is None:
            out[name] = [child.save() for child in self.children]
        else:
            out[name].append({"body": [child.save() for child in self.children]})
            # out["body"] = [child.save() for child in self.children]
        return out


class WhileLoopAST(ScopeAST):
    def __init__(
        self, root: Node | None = None, children: list | None = None, parent=None
    ):
        super().__init__(root, children, parent)
        self.end_while: int = 0
        if len(self.children) > 0:
            for child in self.children:
                child.parent = self

    def handle(self):
        return self

    def to_dict(self):
        return {"while": [self.condition.save()]}, "while"

    def save(self):
        out, name = self.to_dict()
        if out[name] is None:
            out[name] = []
        if self.condition is None:
            out[name] = [child.save() for child in self.children]
        else:
            out[name].append({"body": [child.save() for child in self.children]})
        return out

    def llvm_block1(self, out, index, blocks):
        name = blocks["1"]
        out += f"{name}: ; preds = %{blocks['2']}, %0\n"
        index += 1

        # DFS the condition
        visited = []
        not_visited = [self.condition]
        while len(not_visited) > 0:
            current = not_visited.pop()
            if current not in visited:
                visited.append(current)
                for i in current.children:
                    if not isinstance(i, Node):
                        not_visited.append(i)

        # handle everything separately
        for current in visited:
            output = self.visit_llvm_op(current, index)

            out += output[0]
            index = output[1]
        index2 = blocks["2"]
        index3 = blocks["3"]
        out += f"br i1 %{index}, label %{index2}, label %{index3}\n"
        return out, index

    def llvm_block2(self, out, index, blocks):
        name = blocks["2"]
        out += f"{name}: ; preds = %{blocks['1']}\n"

        # DFS
        visited = visited_list_dfs(self.children[0])

        # Handle
        for current in visited:
            output = tuple
            if current.root.value in tokens:
                output = self.visit_llvm_op(current, index)
            else:
                output = current.llvm(True, index)
            out += output[0]
            index = output[1]

        index1 = blocks["1"]
        out += f"br label %{index1}\n"
        return out, index

    @staticmethod
    def llvm_block3(out, index, blocks):
        name = blocks["3"]
        out += f"{name}: ; preds = %{blocks['1']}\n"
        # out += f"\t%ret i32 0\n"
        index += 1
        return out, index

    def llvm(self, scope: bool = False, index: int = 1) -> tuple[Any, Any]:
        self.blocks = {"1": index, "2": index + 1, "3": index + 2}
        index += 2
        out = f"\tbr label %{index}\n\n"
        out, index = self.llvm_block1(out, index, self.blocks)
        out, index = self.llvm_block2(out, index, self.blocks)
        out, index = self.llvm_block3(out, index, self.blocks)
        return out, index

    def mips(self, registers: Registers):
        output_local = f"while_{registers.globalObjects.index}: \n"
        output_global = ""
        output_list = []
        self.register = registers.globalObjects.index
        registers.globalObjects.index += 1
        self.end_while = registers.globalObjects.index
        output = self.children[0].mips(registers)
        output_list += output[2]
        size = 4 * len(output_list)
        size += 4
        out_condition = self.condition.mips(registers)
        output_local += out_condition[0]
        output_local += f"\tbeq $v1, $zero, end_while_{self.end_while}\n"
        output_list += out_condition[2]
        # condition of the while loop
        output_local += output[0]
        output_global += output[1]
        output_local += f"\tj while_{self.register}\n"
        # end of the while loop
        output_local += f"end_while_{self.end_while}: \n"
        registers.globalObjects.index += 1

        return output_local, output_global, []


class CondAST(TermAST):
    def __init__(
        self, root: Node | None = None, children: list | None = None, parent=None
    ):
        super().__init__(root, children, parent)
        self.last_eval = None

    def __repr__(self) -> str:
        return (
            f"root: {{ {self.root} }}, last_eval: "
            f"{self.last_eval.value if isinstance(self.last_eval, Node) else self.last_eval} , children: {self.children}"
        )

    def llvm(self, scope: bool = False, index: int = 1) -> tuple[str, int]:
        pass

    def mips(self, registers: Registers):
        out_local = ""
        out_global = ""
        out_list = []
        if self.last_eval is not None:
            out_local += f"\tli $v1, {self.last_eval}\n"
        else:
            # the condition is still an ExprAST (thus, not evaluated)
            # dfs the condition
            visited = []
            not_visited = [self]
            while len(not_visited) > 0:
                v = not_visited.pop()
                if v not in visited:
                    if v is not self:
                        visited.append(v)
                    if isinstance(v, AST):
                        for child in v.children:
                            not_visited.append(child)
            visited.reverse()
            for current in visited:
                output = current.mips(registers)
                out_local += output[0]
                out_global += output[1]
                out_list += output[2]
            # get the last used register
            last_register = out_list[-1]
            out_local += f"\tmove $v1, ${last_register}\n"

        return out_local, out_global, out_list


class InitAST(DeclrAST):
    def __init__(
        self, root: Node | None = None, children: list | None = None, parent=None
    ):
        super().__init__(root, children, parent)

    def llvm(self, scope: bool = False, index: int = 1) -> tuple[str, int]:
        pass


class BreakAST(InstrAST):
    def __init__(
        self, root: Node | None = None, children: list | None = None, parent=None
    ):
        super().__init__(root, children, parent)

    def llvm(self, scope: bool = False, index: int = 1) -> tuple[str, int]:
        blocks = self.search_blocks()
        name = blocks["3"]
        out = f"br label %{name}"
        return out, index

    def mips(self, registers: Registers):
        parent = self.parent
        found_last = False
        while parent is not None:
            if isinstance(parent, WhileLoopAST | CaseAST):
                break
            parent = parent.parent
        if isinstance(parent, WhileLoopAST):
            end_while_register = parent.end_while
        elif isinstance(parent, CaseAST):
            end_switch_register = parent.parent.end_label
        else:
            # search for the last if/else in the switch
            for child in parent.parent.children:
                if not isinstance(child, IfCondAST) and not isinstance(
                    child, ElseCondAST
                ):
                    found_last = True
                    end_while_register = child.end_register
                    break
            if not found_last:
                raise Exception("Break outside of a loop or switch")

        out = f"\tj end_{'switch' if isinstance(parent, CaseAST) else 'while'}_{end_while_register if isinstance(parent, WhileLoopAST) else end_switch_register}\n"
        return out, "", []


class ContAST(InstrAST):
    def __init__(
        self, root: Node | None = None, children: list | None = None, parent=None
    ):
        super().__init__(root, children, parent)

    def llvm(self, scope: bool = False, index: int = 1) -> tuple[str, int]:
        blocks = self.search_blocks()
        name = blocks["1"]
        out = f"br label %{name}"
        return out, index

    def mips(self, registers: Registers):
        # get the name of its block
        # search for the parent block that's a while
        parent = self.parent
        while True:
            if isinstance(parent, WhileLoopAST):
                break
            parent = parent.parent
        while_register = parent.register
        out = f"\tj while_{while_register}\n"
        return out, "", []


class FuncParametersAST(AST):
    def __init__(
        self,
        root: Node | None = None,
        children: list | None = None,
        parent=None,
        symbol_table: SymbolTable | None = None,
        parameters: list[FuncParameter | None] | None = None,
    ):
        super().__init__(root, children, parent, symbol_table)
        if parameters is None:
            parameters = []
        self.parameters = parameters

    def handle(self):
        return self

    def save(self):
        return [child.save() for child in self.children]


class FuncDeclAST(AST):
    def __init__(
        self,
        root: Node | None = None,
        children: list | None = None,
        parent=None,
        symbol_table: SymbolTable | None = None,
        return_type: str | None = None,
        const: bool = False,
        ptr: bool = False,
        ptr_level: int = 0,
        params=None,
    ):
        super().__init__(
            root,
            children,
            parent,
            symbol_table=SymbolTable() if symbol_table is None else symbol_table,
        )
        self.symbolTable.owner = self
        self.type: str = return_type
        self.const: bool = const
        self.ptr: bool = ptr
        self.ptr_level: int = ptr_level
        if params is None:
            params = []
        self.params = params
        self.has_defaults = []

    def handle(self):
        return self

    def save(self):
        out, name = self.to_dict()
        if out[name] is None:
            out[name] = []
        for child in self.children:
            if isinstance(child, FuncParametersAST):
                out[name].append({"Parameters": child.save()})
        return out

    def save_dot(self, dictionary_function: dict | None = None):
        return f"{'const ' if self.const else ''}{self.type}{'*'*self.ptr_level} {self.root.key} [label=\"{self.root.key}\"]\n"

    def to_dict(self):
        return (
            {
                f"{'const ' if self.const else ''}{self.type}{'*' * self.ptr_level} {self.root.key}": self.root.value
            },
            f"{'const ' if self.const else ''}{self.type}{'*' * self.ptr_level} {self.root.key}",
        )

    def llvm(self, scope: bool = False, index: int = 1) -> tuple[str, int]:
        out = ""
        # Begin
        out += f"define dso_local {get_llvm_type(self.type)} @{self.root.key}"
        # Parameters
        paramaters = self.params
        param_string = ""
        if len(paramaters) > 0:
            for i, param in enumerate(paramaters):
                param_string += f"{get_llvm_type(param.type)} noundef %{param.key}"
                if i + 1 != len(paramaters):
                    param_string += ", "
        out += f" ({param_string})"
        return out, index

    def mips(self, registers: Registers):
        return "", f".globl {self.root.key}\n" if self.root.key == "main" else "", []


class FuncDefnAST(AST):
    def __init__(
        self,
        root: Node | None = None,
        children: list | None = None,
        parent=None,
        symbol_table: SymbolTable | None = None,
        return_type: str | None = None,
        const: bool = False,
        ptr: bool = False,
        ptr_level: int = 0,
        params=None,
    ):
        super().__init__(
            root,
            children,
            parent,
            symbol_table=SymbolTable() if symbol_table is None else symbol_table,
        )
        self.symbolTable.owner = self
        self.type: str = return_type
        self.const: bool = const
        self.ptr: bool = ptr
        self.ptr_level: int = ptr_level
        if params is None:
            params = []
        self.params = params
        self.has_defaults = []
        self.index = 0

    def handle(self):
        return self

    def save(self):
        out, name = self.to_dict()
        if out[name] is None:
            out[name] = []
        for child in self.children:
            if isinstance(child, FuncParametersAST):
                out[name].append({"Parameters": child.save()})
            else:
                out[name].append({"Body": child.save()})
        return out

    def to_dict(self):
        return (
            {
                f"{'const ' if self.const else ''}{self.type}{'*' * self.ptr_level} {self.root.key}": self.root.value
            },
            f"{'const ' if self.const else ''}{self.type}{'*' * self.ptr_level} {self.root.key}",
        )

    def llvm(self, scope: bool = False, index: int = 1) -> tuple[str, int]:
        out = ""
        # Begin
        out += f"\ndefine dso_local {get_llvm_type(self.type)} @{self.root.key}"
        # Parameters
        parameters = self.params
        param_string = ""
        if len(parameters) > 0:
            for i, param in enumerate(parameters):
                param_string += f"{get_llvm_type(param.type)} noundef %{param.key}"
                if i + 1 != len(parameters):
                    param_string += ", "
        out += f" ({param_string})"
        # the rest
        out += " #0 {\n"

        local_index = 1
        if len(parameters) > 0:
            for child in parameters:
                out += f"%{local_index} = alloca {get_llvm_type(child.type)}, align {'4' if not child.ptr else '8'}\n"
                out += f"%store {get_llvm_type(child.type)} %{child.key}, ptr %{local_index}, align {'4' if not child.ptr else '8'}\n"
                entry, length = self.get_entry(child)
                entry.register = local_index
                local_index += 1
        # Scope
        output = self.children[0].llvm(True, local_index)
        out += output[0]
        out += "\n"
        return out, index

    def mips(self, registers: Registers):
        out_local = f"{self.root.key}:\n"
        out_global = f".globl {self.root.key}\n" if self.root.key == "main" else ""
        out_global += f"{'jal main' if self.root.key == 'main' else ''}\n"
        # Begin
        # Parameters
        for param in self.params:
            if param.type == "float":
                # if f"flt_{param.key}" not in registers.globalObjects.data[1].values():
                #     registers.globalObjects.data[1][0.0] = f"flt_{param.key}"
                if registers.search(param) is None:
                    registers.floatManager.LRU(param)
                if f"flt_{param.key}" not in registers.globalObjects.uninitialized[1]:
                    registers.globalObjects.uninitialized[1].append(f"flt_{param.key}")
                if param.ptr:
                    temp_param = param.value
                    while temp_param.ptr:
                        # declare it in the global scope
                        if (
                            f"int_{temp_param.key}"
                            not in registers.globalObjects.uninitialized[1]
                        ):
                            registers.globalObjects.uninitialized[2].append(
                                f"int_{temp_param.key}"
                            )
                        if temp_param.value is None:
                            break
                        temp_param = temp_param.value
            elif param.type == "int":
                # if f"int_{param.key}" not in registers.globalObjects.data[2].values():
                #     registers.globalObjects.data[2][0] = f"int_{param.key}"
                if registers.search(param) is None:
                    registers.savedManager.LRU(param)
                if f"int_{param.key}" not in registers.globalObjects.uninitialized[2]:
                    registers.globalObjects.uninitialized[2].append(f"int_{param.key}")
                if param.ptr:
                    temp_param = param.value
                    while temp_param.ptr:
                        # declare it in the global scope
                        if (
                            f"int_{temp_param.key}"
                            not in registers.globalObjects.uninitialized[2]
                        ):
                            registers.globalObjects.uninitialized[2].append(
                                f"int_{temp_param.key}"
                            )
                        if temp_param.value is None:
                            break
                        temp_param = temp_param.value
            elif param.type == "char":
                # if f"chr_{param.key}" not in registers.globalObjects.data[4].values():
                #     registers.globalObjects.data[4][0] = f"chr_{param.key}"
                if registers.search(param) is None:
                    registers.savedManager.LRU(param)
                if f"chr_{param.key}" not in registers.globalObjects.uninitialized[4]:
                    registers.globalObjects.uninitialized[4].append(f"chr_{param.key}")
                if param.ptr:
                    temp_param = param.value
                    while temp_param.ptr:
                        # declare it in the global scope
                        if (
                            f"int_{temp_param.key}"
                            not in registers.globalObjects.uninitialized[4]
                        ):
                            registers.globalObjects.uninitialized[2].append(
                                f"int_{temp_param.key}"
                            )
                        if temp_param.value is None:
                            break
                        temp_param = temp_param.value
        # Body
        out_l, out_g, out_list = self.children[0].mips(registers)
        registers.globalObjects.index += 1
        out_local += out_l + "\n"
        out_global += out_g
        return out_local, out_global, out_list


class FuncCallAST(AST):
    def __init__(
        self,
        root: Node | None = None,
        children: list | None = None,
        parent=None,
        symbol_table: SymbolTable | None = None,
        args: list | None = None,
    ):
        super().__init__(root, children, parent, symbol_table)
        if args is None:
            args = []
        self.args = args

    def handle(self):
        return self

    def save(self):
        out, name = self.to_dict()
        if out[name] is None:
            out[name] = []
        out[name].append({"parameters": [child.save() for child in self.children]})
        return out

    def llvm(self, scope: bool = False, index: int = 1) -> tuple[str, int]:
        # %result = call i32 @add(i32 3, i32 4)
        function, length = self.get_entry(self.root)
        # arguments
        arg_string = ""
        count = 0
        for arg in self.args:
            arg_string += f"{get_llvm_type(get_type(arg.value) if isinstance(arg, Node) else arg.type)} {arg.value!s}"
            if count + 1 != len(arg):
                arg_string += ", "
        # end string
        out = f"call {get_llvm_type(function.type)} @{self.root.key}({arg_string})\n"
        return out, index

    def mips(self, registers: Registers):
        out = ""
        temp_list = []
        # load the values of the arguments on the data block
        entry = None
        current = None
        temp_parent = self.parent
        # find the nearest table
        while temp_parent is not None:
            current = temp_parent.symbolTable
            if current is not None:
                break
            temp_parent = temp_parent.parent
        if current is None:
            raise Exception("No symbol table found")
        # go up the table ladder
        while current.parent is not None:
            current = current.parent
        while True:
            for i in current.table:
                if i.object.key == self.root.key:
                    entry = i
                    break
            if current.parent is None:
                break
            current = current.parent
        parameters_org = entry.parameters
        variables = []
        calc = ""
        for index, arg in enumerate(self.args):
            if arg.register is None:
                if isinstance(arg, AST):
                    output = arg.mips(registers)
                    variables += arg.variable_check()
                    calc += output[0]
                    # out += output[0]
                    temp_list += output[2]
                    # parameters_org[count].update(arg.register, registers)
                else:
                    if registers.search(arg) is not None:
                        pass
                    else:
                        registers.savedManager.LRU(arg)
                    if isinstance(arg, VarNode):
                        # if not arg.ptr:
                        variables.append(arg)

            par_type = parameters_org[index].type
            if par_type == "float":
                par_type = "flt_"
            elif par_type == "int":
                par_type = "int_"
            elif par_type == "char":
                par_type = "chr_"
            if parameters_org[index].ptr:
                calc += f"\tsw{'c1' if parameters_org[index].type == 'float' else ''} ${arg.register.name}, {par_type}{parameters_org[index].name}\n"
            else:
                calc += f"\tsw{'c1' if parameters_org[index].type == 'float' else ''} ${arg.register.name}, {par_type}{parameters_org[index].name}\n"
            # out += f"\tmov{'.s' if parameters_org[count].type == 'float' else 'e'} ${arg.register.name}, ${parameters_org[count].object.register.name}\n"
        size = len(variables) * 4
        if size != 0:
            out += f"\taddi $sp, $sp, -{size}\n"
            registers.globalObjects.stackSize += size
        for index, vars in enumerate(variables):
            output = vars.mips(registers)
            out += output[0]
            temp_list += output[2]
            if parameters_org[index].ptr or parameters_org[index].reference:
                continue
            # retrieve value globaly and store it locally
            if registers.search(vars) is None:
                if vars.type == "float":
                    registers.floatManager.LRU(vars)
                else:
                    registers.savedManager.LRU(vars)
            type_string = ""
            if vars.type == "float":
                type_string = "flt"
            elif vars.type == "int":
                type_string = "int"
            elif vars.type == "char":
                type_string = "chr"
            out += f"\tlw{'c1' if vars.type == 'float' else ''} ${vars.register.name}, {type_string}_{vars.value if not isinstance(vars, VarNode) else vars.key}\n"
            out += f"\tsw{'c1' if vars.type == 'float' else ''} ${vars.register.name}, {index * 4}($sp)\n"
            vars.register.shuffle()
        out += calc
        out += f"\tjal {self.root.key}\n"
        for index, vars in enumerate(variables):
            if parameters_org[index].ptr or parameters_org[index].reference:
                out_type = ""
                if vars.type == "float":
                    out_type = "flt"
                elif vars.type == "int":
                    out_type = "int"
                elif vars.type == "char":
                    out_type = "chr"
                if vars.ptr:
                    out += f"\tlw{'c1' if vars.type == 'float' else ''} ${vars.value.register.name}, {out_type}_{parameters_org[index].name}\n"
                    out += f"\tla{'c1' if vars.type == 'float' else ''} ${vars.register.name}, 0(${vars.value.register.name})\n"
                    out += f"\tsw{'c1' if vars.type == 'float' else ''} ${vars.register.name}, {out_type}_{vars.value.key}\n"
                else:
                    out += f"\tlw{'c1' if vars.type == 'float' else ''} ${vars.register.name}, {out_type}_{parameters_org[index].name}\n"
                    out += f"\tsw{'c1' if vars.type == 'float' else ''} ${vars.register.name}, {out_type}_{vars.key}\n"
                continue
            out += f"\tlw{'c1' if vars.type == 'float' else ''} ${vars.register.name}, {index * 4}($sp)\n"
            vars.register.shuffle()
            type_string = ""
            if vars.type == "float":
                type_string = "flt"
            elif vars.type == "int":
                type_string = "int"
            elif vars.type == "char":
                type_string = "chr"
            out += f"\tsw{'c1' if vars.type == 'float' else ''} ${vars.register.name}, {type_string}_{vars.value if not isinstance(vars, VarNode) else vars.key}\n"

        if size != 0:
            out += f"\taddi $sp, $sp, {size}\n"
            registers.globalObjects.stackSize -= size
        registers.search(self.root)
        if self.register is None:
            registers.temporaryManager.LRU(self.root)
            self.register = self.root.register
        out += f"\tmove ${self.register.name}, $v0\n"
        new_node = Node(self.root.key, None)
        new_node.register = self.register
        self.parent.children[self.parent.children.index(self)] = new_node
        return out, "", temp_list


class FuncScopeAST(AST):
    def __init__(
        self,
        root: Node | None = None,
        children: list | None = None,
        parent=None,
        symbol_table: SymbolTable | None = None,
    ):
        super().__init__(root, children, parent, SymbolTable())
        self.symbolTable.owner = self.root.key

    def handle(self):
        return self

    def llvm(self, scope: bool = False, index: int = 1) -> tuple[str, int]:
        # DFS the whole scope and llvm every part of them
        visited = []
        has_return = False
        not_visited = [self]
        while len(not_visited) > 0:
            current = not_visited.pop()
            if current not in visited:
                if current is not self:
                    visited.append(current)
                if not (
                    isinstance(
                        current,
                        WhileLoopAST
                        | FuncDeclAST
                        | IfCondAST
                        | ArrayElementAST
                        | FuncDefnAST,
                    )
                    or (isinstance(current, FuncScopeAST) and current.children)
                ):
                    for i in current.children:
                        if not isinstance(i, Node):
                            not_visited.append(i)

        out = ""
        visited.reverse()
        for current in visited:
            if isinstance(current, ReturnInstr):
                has_return = True
            output = tuple
            if current.root.value in tokens:
                output = self.visit_llvm_op(current, index)
            else:
                output = current.llvm(True, index)
            out += output[0]
            index = output[1]
        if not has_return:
            ret_type = get_llvm_type(self.parent.type)
            out += f"ret {ret_type if self.parent.type != 'void' else ''} {0 if self.parent.type != 'void' else '' } \n"
        out += "}\n"
        return out, index

    def stack_size(self):
        if not self.symbolTable or not self.parent or not self.parent.symbolTable:
            return 0
        size = 0
        # calculate stack size of the function
        for entry in self.symbolTable.table:
            if isinstance(entry.object, VarNode):
                if entry.object.ptr:
                    size += 4
                if entry.object.array:
                    size += 4 * entry.size
                else:
                    size += 4

        # calculate stack size of the function parameters
        for entry in self.parent.symbolTable.table:
            if isinstance(entry.object, VarNode):
                if entry.object.ptr:
                    size += 4
                if entry.object.array:
                    size += 4 * entry.size
                else:
                    size += 4
        return size

    def mips(self, registers: Registers):
        # Begin
        out_global = ""
        # out_local = f"\taddi $sp, $sp, -{size}\n"
        # out_local += "\tsw $ra, 4($sp)\n"
        # out_local = "\tjal allocate_stack\n"
        out_local = ""
        self.parent.index = registers.globalObjects.index
        registers.globalObjects.index += 1
        # DFS
        visited = []
        not_visited = [self]
        while len(not_visited) > 0:
            current = not_visited.pop()
            if current not in visited:
                if current is not self:
                    visited.append(current)
                if not (
                    isinstance(
                        current,
                        WhileLoopAST
                        | FuncDeclAST
                        | IfCondAST
                        | FuncDefnAST
                        | SwitchAST,
                    )
                ):
                    if isinstance(current, Node):
                        continue
                    for entry in current.children:
                        if isinstance(current, InstrAST) and isinstance(
                            entry, ArrayDeclAST
                        ):
                            continue
                        if not (
                            isinstance(entry, Node)
                            and not isinstance(entry.parent, ArrayNode)
                        ):
                            not_visited.append(entry)
        visited.reverse()
        temp_list = []
        out_temp_global = ""
        out_temp_local = ""
        # registers for each entry in the symbol table
        for entry in self.symbolTable.table:
            if isinstance(entry.object, VarNode):
                # if entry.object.ptr:
                #     registers.temporaryManager.LRU(entry.object)
                #     out_temp_local += f"\taddi ${entry.object.register.name}, $sp, 8\n"
                # elif entry.object.array:
                #     registers.temporaryManager.LRU(entry.object)
                #     out_temp_local += f"\taddi ${entry.object.register.name}, $sp, {4 * entry.size}\n"
                # else:
                #     registers.temporaryManager.LRU(entry.object)
                #     out_temp_local += f"\taddi ${entry.object.register.name}, $sp, 4\n"
                # temp_list.append(entry.object.register.name)
                if entry.object.ptr:
                    continue
                if entry.type == "int":
                    if entry.object.key in registers.globalObjects.data[2].values():
                        continue
                    # declare the variable in the global scope .data
                    if entry.object.value is not None:
                        registers.globalObjects.data[2][entry.object.value] = (
                            f"{entry.type}_{entry.object.key}"
                        )
                    else:
                        # if entry.array:
                        #     if len(entry.object.values) == 0:
                        #         registers.globalObjects.uninitialized[3].append(entry.object)
                        #     else:
                        #         registers.globalObjects.uninitialized[3].append(entry.object)
                        # else:
                        if (
                            f"int_{entry.object.key}"
                            in registers.globalObjects.uninitialized[2]
                        ):
                            continue
                        registers.globalObjects.uninitialized[2].append(
                            f"int_{entry.object.key}"
                        )
                elif entry.type == "float":
                    if entry.object.key in registers.globalObjects.data[1].values():
                        continue
                    # declare the variable in the global scope .data
                    if entry.object.value is not None:
                        registers.globalObjects.data[1][entry.object.value] = (
                            f"flt_{entry.object.key}"
                        )
                    else:
                        if entry.array:
                            if (
                                f"flt_{entry.object.key}"
                                in registers.globalObjects.uninitialized[3]
                            ):
                                continue
                            registers.globalObjects.uninitialized[3].append(
                                entry.object
                            )
                        else:
                            if (
                                f"flt_{entry.object.key}"
                                in registers.globalObjects.uninitialized[1]
                            ):
                                continue
                            registers.globalObjects.uninitialized[1].append(
                                f"flt_{entry.object.key}"
                            )
                elif entry.type == "char":
                    if entry.object.key in registers.globalObjects.data[0].values():
                        continue
                    # declare the variable in the global scope .data
                    if entry.object.value is not None:
                        registers.globalObjects.data[0][entry.object.value] = (
                            f"chr_{entry.object.key}"
                        )
                    else:
                        if entry.array:
                            if (
                                f"chr_{entry.object.key}"
                                in registers.globalObjects.uninitialized[3]
                            ):
                                continue
                            registers.globalObjects.uninitialized[3].append(
                                entry.object
                            )
                        else:
                            if (
                                f"chr_{entry.object.key}"
                                in registers.globalObjects.uninitialized[0]
                            ):
                                continue
                            registers.globalObjects.uninitialized[0].append(
                                f"chr_{entry.object.key}"
                            )
        param_str = ""
        # initialize the registers for the parameters
        # registers.temporaryManager.clear()
        for param in self.parent.params:
            if registers.search(param) is not None:
                pass
            elif param.type == "float":
                registers.floatManager.LRU(param)
            else:
                registers.savedManager.LRU(param)
            type_str = ""
            if param.type == "int":
                type_str = "int"
            elif param.type == "float":
                type_str = "flt"
            elif param.type == "char":
                type_str = "chr"
            param_str += f"\tlw{'c1' if type_str == 'float' else ''} ${param.register.name}, {type_str}_{param.key}\n"
        # mips code for each instruction
        for current in visited:
            output = tuple
            if isinstance(current, Node):
                output = current.mips(registers)
            elif current.root.value in tokens:
                output = self.visit_mips_op(current, registers)
            else:
                output = current.mips(registers)
            out_temp_local += output[0]
            out_global += output[1]
            temp_list += output[2]
        # remove duplicates from the list
        temp_list = list(dict.fromkeys(temp_list))
        temp_list.append("ra")
        size = len(temp_list) * 4
        # check if temp_list has v0 if so delete it
        if "v0" in temp_list:
            # delete v0 from the list
            temp_list.remove("v0")
        if "v1" in temp_list:
            temp_list.remove("v1")
        for param in self.parent.params:
            registers.search(param)
            if param.register.name in temp_list:
                temp_list.remove(param.register.name)
        # save the registers
        out_local += f"\taddi $sp, $sp, -{size}\n"
        registers.globalObjects.stackSize += size

        for index, entry in enumerate(temp_list):
            reg_object = registers.searchRegister(entry).object
            if reg_object is not None:
                if isinstance(reg_object, AST):
                    reg_object = f" {reg_object.root.get_str()}"
                elif reg_object != Node("", None):
                    reg_object = f" {reg_object.get_str()}"
                else:
                    reg_object = None
            if entry.startswith("f"):
                out_local += f"\tswc1 ${entry}, {index * 4}($sp)\n"
            else:
                out_local += f"\tsw ${entry}, {index * 4}($sp)\t{'#' + reg_object if reg_object is not None else ''}\n"

        out_local += param_str
        out_local += out_temp_local
        out_global += out_temp_global
        for parm in self.parent.params:
            if parm.ptr or parm.reference:
                # check param type
                type_str = ""
                if parm.type == "int":
                    type_str = "int"
                elif parm.type == "float":
                    type_str = "flt"
                elif parm.type == "char":
                    type_str = "chr"
                out_local += f"\tsw{'' if parm.type != 'float' else 'c1'} ${parm.register.name}, {type_str}_{parm.key}\n"
        # restore registers
        out_local += f"exit_{self.parent.index}:\n"
        for index, entry in enumerate(temp_list):
            if entry.startswith("f"):
                out_local += f"\tlwc1 ${entry}, {index * 4}($sp)\n"
            else:
                out_local += f"\tlw ${entry}, {index * 4}($sp)\n"

        out_local += f"\taddi $sp, $sp, {size}\n"
        registers.globalObjects.stackSize -= size
        # End
        out_local += (
            "\tjr $ra\n"
            if self.parent.root.key != "main"
            else "\tli $v0, 10\n\tsyscall\n"
        )
        return out_local, out_global, []


class ReturnInstr(InstrAST):
    def __init__(
        self, root: Node | None = None, children: list | None = None, parent=None
    ):
        super().__init__(root, children, parent)

    def handle(self):
        return self

    def llvm(self, scope: bool = False, index: int = 1) -> tuple[str, int]:
        out = ""
        temp_type = ""
        child = self.children[0]
        if isinstance(child, Node):
            temp_type = get_llvm_type(child.key)
            if temp_type == "float":
                temp_type = "double"
            out += f"ret {temp_type if child.key is not None else 'i32'} {child.value if child.value is not Node else 0}\n"
        elif isinstance(child, VarNode):
            temp_type = get_llvm_type(child.key)
            entry, length = self.get_entry(child)
            out += f"%{index} = load {temp_type}, {temp_type} %{entry.register}, align {'4' if not child.ptr else '8'}\n"
            out += f"ret {temp_type} %{entry.register}\n"
        return out, index

    def mips(self, registers: Registers):
        out = ""
        if len(self.children) == 0:
            current = self.parent
            while isinstance(current, FuncDefnAST) is False:
                current = current.parent
            out += f"\tj exit_{current.index}\n"
            return out, "", []
        child = self.children[0]
        if isinstance(child, VarNode):
            if child.register is None:
                registers.temporaryManager.LRU(child)
            out += (
                f"\tlw{'c1' if child.type == 'float' else ''} ${child.register.name},"
            )
            if (child.type is not None and child.key == "var") or isinstance(
                child, FuncParameter
            ):
                out += f"{child.type}_{child.key}\n"
            else:
                out += f"{child.value}\n"
            out += f"\tmove $v0, ${child.register.name}\n"
            # out += "\tjr $ra\n"
        else:
            return_value = None
            if child.key == "var":
                type_ = None
                if child.type == "int":
                    type_ = "int"
                elif child.type == "char":
                    type_ = "chr"
                elif child.type == "float":
                    type_ = "flt"
                return_value = f"{type_}_{child.value}"

            elif isinstance(child.register, Register):
                return_value = child.register.name
                out += f"\tmove $v0, ${return_value}\n"
            else:
                return_value = child.value
                out += f"\tli $v0, {return_value}\n"
            # out += "\tjr $ra\n"
        current = self.parent
        while isinstance(current, FuncDefnAST) is False:
            current = current.parent
        out += f"\tj exit_{current.index}\n"
        # out += f"\tj exit_{self.parent.index}\n"
        # out += "\tli $v0, 17\n\tsyscall\n"
        # elif isinstance(child, VarNode):
        #     entry, length = self.getEntry(child)
        #     out += f"\tlw $v0, {entry.offset}($sp)\n"
        return out, "", ["v0"]


class ScanfAST(AST):
    def __init__(
        self,
        root: Node | None = None,
        children: list | None = None,
        parent=None,
        symbol_table: SymbolTable | None = None,
    ):
        super().__init__(root, children, parent, symbol_table)
        self.variables = []
        self.format_string = None
        self.format_specifiers = []
        self.width: int = 0
        self.index: int = 0

    def save(self):
        out, name = self.to_dict()
        if out[name] is None:
            out[name] = []
        if self.root.value is None:
            out[name] = [child.save() for child in self.variables]
        return out

    def to_dict(self):
        name = f"scanf({self.format_string})"
        return {name: self.variables}, name

    def handle(self):
        return self

    def llvm_global(self, index: int = 1) -> tuple[str, int]:
        out = ""
        out += f'@.str.{index} = private unnamed_addr constant [{len(self.format_string)} x i8] c"{self.format_string}\\00", align 1\n'
        entry, length = self.get_entry(self.root)
        entry.register = index
        index += 1
        return out, index

    def llvm(self, scope: bool = False, index: int = 1) -> tuple[str, int]:
        out = ""
        var_string = ""
        count = 0
        for var in self.variables:
            if isinstance(var, Node):
                var_string += (
                    f"{get_llvm_type(get_type(var.value))} noundef %{var.value}"
                )
            else:
                entry, length = self.get_entry(var)
                var += f"{get_llvm_type(entry.type)} noundef %{entry.register}"
            if count + 1 != len(self.variables):
                var_string += ", "
        out += f"call i32 (ptr, ...) @__isoc99_scanf(ptr noundef @.str.{index}, {var_string})\n"
        return out, index

    def format(self):
        format_ = re.split(r"(%[0-9]*[discf])|(\\\\0A)", str(self.format_string))
        return [x for x in format_ if x is not None and x != ""]

    def mips(self, registers: Registers):
        # Scanf in mips
        # format the format string.
        # Ask for input depending on the format string, so different syscall for different types
        # store the input in the variables (registers that are assigned to the variables)
        # check if the completed format string is correct
        # if not, print an error message and exit
        # if yes, continue
        out_local = ""
        out_reg = []
        format_ = self.format()
        for var in self.variables:
            if registers.temporaryManager.search(var) is None:
                # assign registers to variables
                if var.type == "float":
                    registers.floatManager.LRU(var)
                else:
                    registers.temporaryManager.LRU(var)
            out_reg.append(var.register)

        # if self.format_string not in registers.globalObjects.data[0].keys():
        #     registers.globalObjects.data[0][self.format_string] = f"format_{self.format_string}"
        # out_local += f"\tla $a0, format_{self.format_string}\n"
        # out_local += f"\tla $a1, input_buffer\n"
        # out_local += f"\tli $a2, 100\n"
        # out_local += f"\tli $v0, 8\n"
        # out_local += f"\tsyscall\n"
        #
        # out_local += f"\tla $a0, format_{self.format_string}\n"
        # out_local += f"\tla $a1, input_buffer\n"
        # for i in range(len(format_)):
        #     if format_[i][0] == '%':
        #         if format_[i][-1] == "d":
        out_list = []
        for index, format_str in enumerate(format_):
            if format_str.startswith("%"):
                if format_str.endswith("d"):
                    out_local += "\tli $v0, 5\n"
                elif format_str.endswith("f"):
                    out_local += "\tli $v0, 6\n"
                elif format_str.endswith("c"):
                    out_local += "\tli $v0, 12\n"
                elif format_str.endswith("s"):
                    type_ = ""
                    if self.variables[index].type == "int":
                        type_ = "int"
                    elif self.variables[index].type == "float":
                        type_ = "flt"
                    elif self.variables[index].type == "char":
                        type_ = "chr"
                    out_local += f"\tla $a0, {type_}_{self.variables[index].key}\n"
                    format_string = ""
                    if format_str[1:-1].isdigit():
                        out_local += f"\tli $a1, {int(format_str[1:-1]) + 1}\n"
                    if format_str not in registers.globalObjects.data[0]:
                        registers.globalObjects.data[0][format_str] = (
                            f"format_{format_str[1:]}"
                        )
                        format_string = f"format_{format_str[1:]}"
                    else:
                        format_string = registers.globalObjects.data[0][format_str]
                    out_local += f"\tla $a2, {format_string}\n"
                    out_local += "\tli $v0, 8\n"
                elif format_str.endswith("i"):
                    out_local += "\tli $v0, 5\n"
                else:
                    out_local += "\tli $v0, 5\n"
                out_local += "\tsyscall\n"
                out_list.append("v0")
                variable_register = self.variables[index].register.name
                if format_str.endswith("s"):
                    pass
                else:
                    out_local += f"\tmov{'.s' if variable_register[0] == 'f' else 'e'} ${variable_register}, $v0\n"
                    out_local += f"\tsw{'c1' if self.variables[index].type == 'float' else ''} $v0, {self.variables[index].type}_{self.variables[index].key}\n"
                    out_list.append(variable_register)
        out_list = list(dict.fromkeys(out_list))
        return out_local, "", out_list


class ArrayDeclAST(AST):
    def __init__(
        self,
        root: Node | None = None,
        children: list | None = None,
        parent=None,
        symbol_table: SymbolTable | None = None,
        size: int = 0,
        ptr_size: int = 0,
        arr_type: str | None = None,
        values=None,
    ):
        super().__init__(root, children, parent, symbol_table)
        if values is None:
            values = []
        self.size = size
        self.values = values
        self.ptr_size = ptr_size
        self.type = arr_type
        self.stack_indexes = []  # indexes of the array that are stored on the stack

    def handle(self):
        return self

    def save(self):
        if len(self.children) > 0:
            if isinstance(self.children[0], ArrayNode):
                return self.children[0].save()
            return super().save()
        return super().save()

    def llvm_global(self, index: int = 1) -> tuple[str, int]:
        out = ""
        entry, length = self.get_entry(self.root)
        out += (
            f"@__const.{entry.symbol_table.owner}.{entry.name} = private unnamed_addr constant "
            f"[{self.size if self.size > 1 else 1} x {get_llvm_type(get_type(self.root.value))}] ["
        )
        count = 0
        vals = ""
        for val in self.values:
            vals += (
                f"{get_llvm_type(entry.type)} {val.value} "
                f"{', ' if count + 1 != len(self.values) else ''}"
            )
            count += 1
        if count < self.size != 0:
            for _i in range(self.size - count):
                vals += f"{get_llvm_type(entry.type)} 0 {', ' if count + 1 != len(self.values) else ''}"
        out += vals
        out += f"], align {4 if self.size < 4 else 16}\n"
        return out, index

    def llvm(self, scope: bool = False, index: int = 1) -> tuple[str, int]:
        out = ""
        entry, length = self.get_entry(self.root)
        if scope:
            # local
            out += f"%{index} = alloca [ {self.size} x {get_llvm_type(entry.type)}], align {4 if self.size < 4 else 16}\n"
            entry.register = index
            out += (
                f"call void @llvm.memcpy.p0.p0.i64(ptr allign {4 if self.size < 4 else 16} %{index}, "
                f"ptr align {4 if self.size < 4 else 16} "
                f"@__const.{entry.symbol_table.owner}.{entry.name}, i64 {self.size * 4}, i1 false)"
            )
            out += "\n"
            index += 1

        else:
            # global
            vals = "["
            count = 0
            for val in self.values:
                vals += f"{get_llvm_type(get_type(entry.type))} {val.value} {', ' if count + 1 != len(self.values) else ''}"
                count += 1
            if count < self.size != 0:
                for _i in range(self.size - count):
                    vals += f"{get_llvm_type(get_type(entry.type))} 0 {', ' if count + 1 != len(self.values) else ''}"
            vals += "]"
            out += (
                f"@{self.root.key} = dso_local global [ "
                f"{self.size if self.size > 1 else 1} x {get_llvm_type(get_type(entry.type))}] "
                f"{'zeroinitializer' if len(self.values) == 0 else vals}, align {4 if self.size < 4 else 16}\n"
            )
        return out, index

    def mips(self, registers: Registers):
        glb = self.global_check()
        out_local = ""
        out_global = ""
        out_list = []
        if glb:
            # make use of data segment to make global variables
            # if values are defined in the array
            # check type first
            if (
                self.type == "int"
                and self.root.key not in registers.globalObjects.data[2].values()
            ):
                temp_string = ""
                for i in range(len(self.values)):
                    temp_string += f"{self.values[i].value}"
                    if i != len(self.values) - 1:
                        temp_string += ", "
                if len(self.values) < self.size:
                    temp_string += ", 0" * (self.size - len(self.values))
                registers.globalObjects.data[2][temp_string] = (
                    f"{self.type}_{self.root.key[:-2]}"  # remove the [] from the key
                )
            elif (
                self.type == "float"
                and self.root.key not in registers.globalObjects.data[1].values()
            ):
                temp_string = ""
                for i in self.values:
                    temp_string += f"{i.value}"
                    if i != self.values[-1]:
                        temp_string += ", "
                if len(self.values) < self.size:
                    temp_string += ", 0.0" * (self.size - len(self.values))
                registers.globalObjects.data[1][temp_string] = (
                    f"{self.type}_{self.root.key[:-2]}"
                )
            elif (
                self.type == "char"
                and self.root.key not in registers.globalObjects.data[4].values()
            ):
                temp_string = ""
                for i in self.values:
                    temp_string += f"'{i.value}'"
                    if i != self.values[-1]:
                        temp_string += ", "
                registers.globalObjects.data[4][temp_string] = (
                    f"{self.type}_{self.root.key[:-2]}"
                )
                if len(self.values) < self.size:
                    temp_string += ", 0" * (self.size - len(self.values))
        else:
            # make use of stack to make local variables
            # allocate memory for the array
            # out_local += f"\taddi $sp, $sp, -{self.size * 4}\n"
            # store the values of the array in the stack
            temp_node = Node("temp", None, None)
            if self.type == "float":
                registers.floatManager.LRU(temp_node)
            else:
                registers.temporaryManager.LRU(temp_node)
            if registers.search(self.root) is None:
                if self.root.type == "float":
                    registers.floatManager.LRU(self.root)
                else:
                    registers.temporaryManager.LRU(self.root)
            type_ = ""
            if self.root.type == "int":
                type_ = "int"
            elif self.root.type == "float":
                type_ = "flt"
            elif self.root.type == "char":
                type_ = "chr"
            out_local += (
                f"\tla ${self.root.register.name}, {type_}_{self.root.key[0]}\n"
            )
            if self.root.register.name.startswith("f"):
                registers.floatManager.shuffle_name(self.root.register.name)
            else:
                registers.temporaryManager.shuffle_name(self.root.register.name)
            for i in range(len(self.values)):
                out_local += f"\tli{'.s' if self.type == 'float' else ''} ${temp_node.register.name}, {self.values[i].value}\n"
                temp_node.register.shuffle()
                if self.root.type == "float" or self.root.type == "int":
                    out_local += f"\taddi ${self.root.register.name}, ${self.root.register.name}, {0 if i == 0 else 4}\n"
                    out_local += f"\tsw{'c1' if self.type == 'float' else ''} ${temp_node.register.name}, 0(${self.root.register.name})\n"
                    self.root.register.shuffle()
                else:
                    out_local += f"\taddi ${self.root.register.name}, ${self.root.register.name}, {0 if i == 0 else 1}\n"
                    out_local += f"\tsb ${temp_node.register.name}, 0(${self.root.register.name})\n"
                    self.root.register.shuffle()
                # out_local += f"\tmov{'.s' if self.type == 'float' else 'e'} ${self.root.register.name}, ${temp_node.register.name}\n\n"
                self.stack_indexes.append((i * 4) + registers.globalObjects.stackSize)
            registers.globalObjects.stackSize += self.size * 4
            out_list.append(temp_node.register.name)
            if self.type == "float":
                registers.floatManager.LRU_delete(temp_node.register.name)
            else:
                registers.temporaryManager.LRU_delete(temp_node.register.name)
        return out_local, out_global, out_list


class IncludeAST(AST):
    def __init__(
        self,
        root: Node | None = None,
        children: list | None = None,
        parent=None,
        symbol_table: SymbolTable | None = None,
    ):
        super().__init__(root, children, parent, symbol_table)

    def to_dict(self):
        return {
            f"#include<{self.root.key}>": self.root.value
        }, f"#include<{self.root.key}>"

    def handle(self):
        return self

    def llvm(self, scope: bool = False, index: int = 1) -> tuple[str, int]:
        pass

    def llvm_global(self, index: int = 1) -> tuple[str, int]:
        return (
            "declare i32 @printf(ptr noundef, ...) #2\n\ndeclare i32 @__isoc99_scanf(ptr noundef, ...) #2\n\n",
            index,
        )

    def mips(self, registers: Registers):
        # hardcode the printf and scanf functions
        registers.globalObjects.data[5][254] = "buffer"
        return "", "", []


class SwitchAST(AST):
    def __init__(
        self,
        root: Node | None = None,
        children: list | None = None,
        parent=None,
        symbol_table: SymbolTable | None = None,
    ):
        super().__init__(root, children, parent, symbol_table)
        self.cases = []
        self.has_default = False
        self.default = None
        self.condition = None
        self.symbolTable = SymbolTable()
        self.index = 0
        self.end_label = 0

    def mips(self, registers: Registers):
        out_local = out_global = ""
        out_list = []
        # condition
        output = self.condition.mips(registers)
        out_local += output[0]
        out_global += output[1]
        out_list += output[2]
        # switch
        if isinstance(self.condition, AST):
            compare_register = self.condition.root.register.name
        else:
            compare_register = self.condition.register.name
        cases_list = []
        cases_cond_list = []
        self.end_label = registers.globalObjects.index
        registers.globalObjects.index += 1
        for case in self.cases:
            temp_local, temp_global, temp_list = case.condition.mips(registers)
            case.index = registers.globalObjects.index
            registers.globalObjects.index += 1
            temp_local += f"\tbeq ${compare_register}, ${case.condition.root.register.name if isinstance(case.condition, AST) else case.condition.register.name}, case_{case.index}\n"
            if compare_register.startswith("t"):
                registers.temporaryManager.shuffle_name(compare_register)
            else:
                registers.floatManager.shuffle_name(compare_register)
            cases_cond_list.append((temp_local, temp_global, temp_list))
            cases_list.append(case.mips(registers))
        if self.has_default:
            self.default.index = registers.globalObjects.index
            registers.globalObjects.index += 1
            cases_list.append(self.default.mips(registers))
        # make the switch
        self.index = registers.globalObjects.index
        registers.globalObjects.index += 1
        out_local += f"switch_{self.index}:\n"
        for case in cases_cond_list:
            out_local += case[0]
            out_global += case[1]
            out_list += case[2]
        if self.has_default:
            out_local += f"\tj default_{self.default.index}\n"
        for case in cases_list:
            out_local += case[0]
            out_global += case[1]
            out_list += case[2]
        out_local += f"end_switch_{self.end_label}:\n"
        # filter duplicates from the list
        out_list = list(dict.fromkeys(out_list))
        return out_local, out_global, out_list


class SwitchScopeAST(ScopeAST):
    def __init__(
        self,
        root: Node | None = None,
        children: list | None = None,
        parent=None,
        condition: AST | None = None,
    ):
        super().__init__(root, children, parent, condition)


class CaseAST(ScopeAST):
    def __init__(
        self,
        root: Node | None = None,
        children: list | None = None,
        parent=None,
        condition: AST | None = None,
    ):
        super().__init__(root, children, parent, condition)
        self.index = 0

    def mips(self, registers: Registers):
        out_local = out_global = ""
        out_list = []
        # case
        # self.index = registers.globalObjects.index
        # registers.globalObjects.index += 1
        out_local += f"case_{self.index}:\n"
        for child in self.children:
            output = child.mips(registers)
            out_local += output[0]
            out_global += output[1]
            out_list += output[2]
        # out_local += f"\tj end_switch_{self.parent.end_label}\n"
        return out_local, out_global, out_list


class DefaultAST(CaseAST):
    def __init__(
        self,
        root: Node | None = None,
        children: list | None = None,
        parent=None,
        condition: AST | None = None,
    ):
        super().__init__(root, children, parent, condition)

    def mips(self, registers: Registers):
        out_local = out_global = ""
        out_list = []
        # default
        out_local += f"default_{self.index}:\n"
        for child in self.children:
            output = child.mips(registers)
            out_local += output[0]
            out_global += output[1]
            out_list += output[2]
        # out_local += f"\tj end_switch_{self.parent.end_label}\n"
        return out_local, out_global, out_list


class CommentAST(AST):
    def __init__(
        self,
        root: Node | None = None,
        children: list | None = None,
        parent=None,
        symbol_table: SymbolTable | None = None,
    ):
        super().__init__(root, children, parent, symbol_table)
        self.comment = ""

    def handle(self):
        return self

    def mips(self, registers: Registers):
        out_local = out_global = ""
        out_list = []
        self.comment = self.comment.replace("\n", "\n#")
        out_local += f"# {self.comment}\n"
        return out_local, out_global, out_list

    def save(self):
        return None
