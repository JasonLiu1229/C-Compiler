import copy
import re
from decimal import Decimal

import antlr4.tree.Tree

from antlr4_output.FileParser import FileParser
from antlr4_output.FileVisitor import FileVisitor

from .ast import (
    AST,
    ArrayDeclAST,
    ArrayElementAST,
    AssignAST,
    BreakAST,
    CaseAST,
    CommentAST,
    CondAST,
    ContAST,
    DeclrAST,
    DefaultAST,
    DerefAST,
    Else_CondAST,
    ExprAST,
    FactorAST,
    For_loopAST,
    FuncCallAST,
    FuncDeclAST,
    FuncDefnAST,
    FuncParametersAST,
    FuncScopeAST,
    If_CondAST,
    IncludeAST,
    InitAST,
    InstrAST,
    PrimaryAST,
    PrintfAST,
    ReturnInstr,
    ScanfAST,
    Scope_AST,
    SwitchAST,
    SwitchScopeAST,
    TermAST,
    VarDeclrAST,
    While_loopAST,
    conv_promotions,
    conversions,
    getType,
    isfloat,
    keywords,
    keywords_datatype,
)
from .node import ArrayNode, FuncParameter, Node, VarNode
from .symbol_entry import (
    FuncSymbolEntry,
    FunctionParameter,
    SymbolEntry,
)
from .symbol_table import SymbolTable


class AstCreator(FileVisitor):
    def __init__(self, filename: str | None = None) -> None:
        """
        Initializer function
        """
        super().__init__()
        self.base_ast: AST = AST()
        self.symbol_table: SymbolTable = SymbolTable()
        self.warnings: list = []
        self.file_name: str | None = filename

    def visit_child(self, ctx):
        """
        visit the right visit function for the give context
        :param ctx: the context to know what to visit
        :return: the given output given by every visit function (AST or Node)
        """
        if isinstance(ctx, FileParser.InstrContext):
            return self.visitInstr(ctx)
        if isinstance(ctx, FileParser.ExprContext):
            return self.visitExpr(ctx)
        if isinstance(ctx, FileParser.RvarContext):
            return self.visitRvar(ctx)
        if isinstance(ctx, FileParser.RtypeContext):
            return self.visitRtype(ctx)
        if isinstance(ctx, FileParser.AssignContext):
            return self.visitAssign(ctx)
        if isinstance(ctx, FileParser.LvarContext):
            return self.visitLvar(ctx)
        if isinstance(ctx, FileParser.DerefContext):
            return self.visitDeref(ctx)
        if isinstance(ctx, FileParser.PrintfContext):
            return self.visitPrintf(ctx)
        if isinstance(ctx, FileParser.Var_declContext):
            return self.visitVar_decl(ctx)
        if isinstance(ctx, FileParser.DeclrContext):
            return self.visitDeclr(ctx)
        if isinstance(ctx, FileParser.TermContext):
            return self.visitTerm(ctx)
        if isinstance(ctx, FileParser.FactorContext):
            return self.visitFactor(ctx)
        if isinstance(ctx, FileParser.PrimaryContext):
            return self.visitPrimary(ctx)
        if isinstance(ctx, FileParser.ScopeContext):
            return self.visitScope(ctx)
        if isinstance(ctx, FileParser.For_loopContext):
            return self.visitFor_loop(ctx)
        if isinstance(ctx, FileParser.While_loopContext):
            return self.visitWhile_loop(ctx)
        if isinstance(ctx, FileParser.If_condContext):
            return self.visitIf_cond(ctx)
        if isinstance(ctx, FileParser.Else_condContext):
            return self.visitElse_cond(ctx)
        if isinstance(ctx, FileParser.InitContext):
            return self.visitInit(ctx)
        if isinstance(ctx, FileParser.CondContext):
            return self.visitCond(ctx)
        if isinstance(ctx, FileParser.IncrContext):
            return self.visitIncr(ctx)
        if isinstance(ctx, FileParser.Cont_instrContext):
            return self.visitCont_instr(ctx)
        if isinstance(ctx, FileParser.Break_instrContext):
            return self.visitBreak_instr(ctx)
        if isinstance(ctx, FileParser.Func_defnContext):
            return self.visitFunc_defn(ctx)
        if isinstance(ctx, FileParser.Func_declContext):
            return self.visitFunc_decl(ctx)
        if isinstance(ctx, FileParser.Arg_listContext):
            return self.visitArg_list(ctx)
        if isinstance(ctx, FileParser.Func_callContext):
            return self.visitFunc_call(ctx)
        if isinstance(ctx, FileParser.Func_scopeContext):
            return self.visitFunc_scope(ctx)
        if isinstance(ctx, FileParser.Func_argContext):
            return self.visitFunc_arg(ctx)
        if isinstance(ctx, FileParser.Param_declrContext):
            return self.visitParam_declr(ctx)
        if isinstance(ctx, FileParser.Param_listContext):
            return self.visitParam_list(ctx)
        if isinstance(ctx, FileParser.Return_instrContext):
            return self.visitReturn_instr(ctx)
        if isinstance(ctx, FileParser.Array_declContext):
            return self.visitArray_decl(ctx)
        if isinstance(ctx, FileParser.Array_elContext):
            return self.visitArray_el(ctx)
        if isinstance(ctx, FileParser.Incl_statContext):
            return self.visitIncl_stat(ctx)
        if isinstance(ctx, FileParser.ScanfContext):
            return self.visitScanf(ctx)
        if isinstance(ctx, FileParser.CompContext):
            return self.visitComp(ctx)
        if isinstance(ctx, FileParser.Switch_instrContext):
            return self.visitSwitch_instr(ctx)
        if isinstance(ctx, FileParser.Case_instrContext):
            return self.visitCase_instr(ctx)
        if isinstance(ctx, FileParser.Default_instrContext):
            return self.visitDefault_instr(ctx)
        if isinstance(ctx, FileParser.Switch_scopeContext):
            return self.visitSwitch_scope(ctx)
        if isinstance(ctx, FileParser.CommentContext):
            return self.visitComment(ctx)
        if isinstance(ctx, antlr4.tree.Tree.TerminalNodeImpl) and ctx.getText() in [
            "{",
            "}",
        ]:
            return Node(ctx.getText(), None)

    @staticmethod
    def searchPrevToken(index: int, in_list, token: str = "}"):
        # index = len(in_list)
        for i in reversed(range(index)):
            if isinstance(in_list[i], Node) and in_list[i].key == token:
                return i
        return -1

    @staticmethod
    def lastInstruction(index: int, in_list, token: str = "}"):
        for i in reversed(range(index)):
            if (
                isinstance(in_list[i], PrintfAST)
                or isinstance(in_list[i], VarDeclrAST)
                or isinstance(in_list[i], AssignAST)
                or isinstance(in_list[i], InstrAST)
                or isinstance(in_list[i], Scope_AST)
                or (isinstance(in_list[i], Node) and in_list[i].key == token)
                or isinstance(in_list[i], FuncDeclAST)
                or isinstance(in_list[i], FuncDefnAST)
                or isinstance(in_list[i], ScanfAST)
                or isinstance(in_list[i], PrintfAST)
                or isinstance(in_list[i], SwitchAST)
                or isinstance(in_list[i], CommentAST)
            ):
                return i
        return -1

    @staticmethod
    def lastDeclaration(index: int, in_list, token: str = "}"):
        for i in reversed(range(index)):
            if (
                isinstance(in_list[i], PrintfAST)
                or isinstance(in_list[i], AssignAST)
                or isinstance(in_list[i], InstrAST)
                or (isinstance(in_list[i], Node) and in_list[i].key == token)
                or isinstance(in_list[i], FuncDeclAST)
                or isinstance(in_list[i], FuncDefnAST)
                or isinstance(in_list[i], DeclrAST)
                or isinstance(in_list[i], Scope_AST)
                or isinstance(in_list[i], ScanfAST)
                or isinstance(in_list[i], PrintfAST)
                or isinstance(in_list[i], SwitchAST)
                or isinstance(in_list[i], CommentAST)
            ):
                return i
        return -1

    @staticmethod
    def lastInit(index: int, in_list):
        for i in reversed(range(index)):
            if isinstance(in_list[i], CondAST):
                return i
        return -1

    @staticmethod
    def lastElse(index: int, in_list):
        for i in reversed(range(index)):
            if isinstance(in_list[i], Else_CondAST):
                return i
        return -1

    @staticmethod
    def lastFuncScope(index: int, in_list):
        for i in reversed(range(index)):
            if isinstance(in_list[i], FuncScopeAST):
                return i
        return -1

    @staticmethod
    def lastDefaultOrCase(index: int, in_list):
        for i in reversed(range(index)):
            if isinstance(in_list[i], CaseAST | DefaultAST):
                return i
            # elif isinstance(in_list[i], BreakAST):
            #     return i - 1
        return -1

    @staticmethod
    def lastSwitchScope(index: int, in_list):
        for i in reversed(range(index)):
            if isinstance(in_list[i], SwitchScopeAST):
                return i
        return -1

    def resolveTree(self, base: AST):
        """
        visit the right visit function for the give context
        :param base: The base AST given to resolve
        :return: the given output given by every visit function (AST or Node)
        """
        # Terminals processing
        index = 0
        indexes = {
            "last_instr": 0,
            "last_declr": 0,
            "last_scope": [0],
            "last_scope_open": 0,
            "scope_depth": 0,
        }
        last_else = None
        for child in base.children[:]:
            if isinstance(child, AST):
                if child.root.key in ["expr", "term"] and child.root.value is not None:
                    if child.root.value in ["++", "--", "!", "const"]:
                        child.children = base.children[index - 1 : index]
                        base.children[index - 1 : index] = []
                        index -= 1
                    else:
                        child.children = base.children[index - 2 : index]
                        base.children[index - 2 : index] = []
                        index -= 2
                    child.children.reverse()

                elif child.root.key == "factor" and child.root.value is not None:
                    if child.root.value in ["++", "+", "--", "-"]:
                        child.children = base.children[index - 1 : index]
                        base.children[index - 1 : index] = []
                        index -= 1
                    child.children.reverse()

                elif child.root.key == "primary" and child.root.value is not None:
                    child.children = base.children[index - 1 : index]
                    base.children[index - 1 : index] = []
                    index -= 1

                elif isinstance(child, ArrayElementAST):
                    child.children = base.children[index - 1 : index]
                    child.root.key = child.children[0].key
                    base.children[index - 1 : index] = []
                    index -= 1
                    if child.root.value is None:
                        child.root.value = base.children[index - 1]
                        child.children.append(base.children[index - 1])
                        base.children[index - 1 : index] = []
                        index -= 1
                elif isinstance(child, CommentAST):
                    child.parent = base
                    index += 1
                    continue
                elif isinstance(child, ArrayDeclAST):
                    child.children = base.children[
                        index - 1 - len(child.values) : index - 1
                    ]
                    child.children.reverse()
                    if len(child.values) > 0:
                        base.children[index - 1 - len(child.values) - 1 : index] = []
                    index = base.children.index(child)
                elif isinstance(child, SwitchAST):
                    last_token = self.searchPrevToken(index, base.children, token="{")
                    child.condition = base.children[last_token + 1]
                    base.children[last_token:index] = []
                    # child.condition.pop(0)
                    # child.condition.reverse()
                    # if len(child.condition) == 1 and isinstance(child.condition[0], Node):
                    #     if child.condition[0].key != "var":
                    #         child.root.value = child.condition[0].value
                    if isinstance(child.condition, Node):
                        if child.condition.key != "var":
                            child.root.value = child.condition.value
                    index = base.children.index(child)
                    last_token = self.searchPrevToken(index, base.children)
                    child.cases = base.children[last_token + 1 : index]
                    # delete the switch token
                    # child.cases.pop(0)
                    child.cases.reverse()
                    child.children = child.cases
                    if isinstance(child.cases[-1], DefaultAST):
                        child.default = child.cases[-1]
                        child.has_default = True
                        child.cases = child.cases[:-1]
                    base.children[last_token:index] = []
                    for i in child.children:
                        i.parent = child
                    index = base.children.index(child)

                elif isinstance(child, DefaultAST):
                    child.children = base.children[index - 1 : index]
                    base.children[index - 1 : index] = []
                    index = base.children.index(child)

                elif isinstance(child, CaseAST):
                    last_switch_scope = self.lastSwitchScope(index, base.children)
                    child.condition = base.children[last_switch_scope + 1 : index]
                    child.condition.reverse()
                    if len(child.condition) == 1 and isinstance(
                        child.condition[0], Node
                    ):
                        if child.condition[0].key != "var":
                            child.root.value = child.condition[0].value
                        child.condition = child.condition[0]
                    base.children[last_switch_scope + 1 : index] = []
                    index = base.children.index(child)
                    child.children = base.children[index - 1 : index]
                    base.children[index - 1 : index] = []
                    index = base.children.index(child)

                elif isinstance(child, SwitchScopeAST):
                    last_case_default = self.lastDefaultOrCase(index, base.children)
                    last_token = self.searchPrevToken(index, base.children)
                    child.children = base.children[
                        max(last_case_default, last_token) + 1 : index
                    ]
                    child.children.reverse()
                    base.children[max(last_case_default, last_token) + 1 : index] = []
                    index = base.children.index(child)

                elif isinstance(child, ScanfAST):
                    child.children = base.children[index - len(child.variables) : index]
                    child.children.reverse()
                    base.children[index - len(child.variables) : index] = []
                    index -= len(child.variables)

                elif isinstance(child, IncludeAST):
                    child.parent = base

                elif isinstance(child, FuncDeclAST):
                    if isinstance(base.children[index - 1], FuncParametersAST):
                        child.children = base.children[index - 1 : index]
                        base.children[index - 1 : index] = []
                    child.children.reverse()
                    if len(child.children) > 0:
                        if isinstance(child.children[0], FuncParametersAST):
                            child.params = child.children[0].parameters
                            child.children = child.children[1:]
                        for param in child.params:
                            if param.value is not None:
                                child.has_defaults.append(param)
                    index = base.children.index(child)
                    child.parent = base

                elif isinstance(child, FuncDefnAST):
                    last_func = self.lastFuncScope(index=index, in_list=base.children)
                    child.children = base.children[last_func:index]
                    base.children[last_func:index] = []
                    child.children.reverse()
                    if isinstance(child.children[0], FuncParametersAST):
                        child.params = child.children[0].parameters
                        child.children = child.children[1:]
                    for param in child.params:
                        if param.value is not None:
                            child.has_defaults.append(param)
                    index = base.children.index(child)
                    child.parent = base

                elif isinstance(child, FuncCallAST):
                    amt = len(child.args)
                    for i in reversed(range(1, amt + 1)):
                        child.args[i - 1] = base.children[index - i]
                    base.children[index - amt : index] = []
                    child.children = child.args
                    index = base.children.index(child)

                elif isinstance(child, ReturnInstr):
                    if child.root.value is None:
                        last_token = (
                            self.searchPrevToken(
                                index=index, token="}", in_list=base.children
                            )
                            + 1
                        )
                        child.children = base.children[index - 1 : index]
                        base.children[last_token:index] = []
                        index = base.children.index(child)

                elif isinstance(child, ContAST) or isinstance(child, BreakAST):
                    last_token = self.searchPrevToken(
                        index=index, token="}", in_list=base.children
                    )
                    last_case_default = self.lastDefaultOrCase(
                        index=index, in_list=base.children
                    )
                    base.children[max(last_case_default, last_token) + 1 : index] = []
                    index = base.children.index(child)

                elif isinstance(child, FuncParametersAST):
                    child.parameters = base.children[
                        index - len(child.parameters) : index
                    ]
                    base.children[index - len(child.parameters) : index] = []
                    child.parameters.reverse()
                    child.children = child.parameters
                    # check default parameters order
                    default_found = False
                    for param in child.parameters:
                        param.parent = child
                        if param.value is not None:
                            default_found = True
                        elif param.value is None and default_found:
                            raise AttributeError(
                                "Default value cannot be followed by non-default value"
                            )
                    index = base.children.index(child)

                elif isinstance(child, CondAST):
                    child.children = base.children[index - 1 : index]
                    base.children[index - 1 : index] = []

                    child.children.reverse()
                    index = base.children.index(child)

                elif isinstance(child, InitAST):
                    last_decl = self.lastInit(index, base.children)
                    last_decl += 1
                    child.children = base.children[last_decl:index]
                    child.children.reverse()
                    base.children[last_decl:index] = []
                    index = base.children.index(child)
                    update_index = last_decl
                    update_index += 1

                elif isinstance(child, InstrAST):
                    # Parent of instr is base itself, if no parent is already found
                    if child.parent is None:
                        child.parent = base
                    if (
                        self.searchPrevToken(
                            index=index, token="}", in_list=base.children
                        )
                        == -1
                    ):
                        last_inst = self.lastInstruction(index, base.children)
                        child.children = base.children[last_inst + 1 : index]
                        base.children[last_inst + 1 : index] = []
                    else:
                        last_inst = self.lastInstruction(index, base.children)
                        child.children = base.children[last_inst + 1 : index]
                        base.children[last_inst + 1 : index] = []
                    child.children.reverse()
                    index = base.children.index(child)
                    indexes["last_instr"] = index + 1

                elif isinstance(child, If_CondAST) or isinstance(child, While_loopAST):
                    if child.parent is None:
                        child.parent = base
                    number = 2
                    else_child = None
                    if isinstance(base.children[index - 3], Else_CondAST):
                        number = 3
                        else_child = base.children[index - 3]
                    child.children = base.children[index - number : index]
                    base.children[index - number : index] = []
                    child.children.reverse()
                    # assign condition
                    child.condition = child.children[0]
                    child.condition.parent = child
                    child.children = child.children[1:]
                    # set the in_loop flag to false for all children in scope if "IF_condition"
                    if isinstance(child, If_CondAST):
                        for ch in child.children[0].children:
                            ch.in_loop = child.in_loop
                    if isinstance(child, While_loopAST):
                        child.condition.in_loop = True
                        for ch in child.children:
                            ch.in_loop = True
                    index = base.children.index(child)

                elif isinstance(child, Else_CondAST):
                    if child.parent is None:
                        child.parent = base
                    child.children = base.children[index - 1 : index]
                    base.children[index - 1 : index] = []
                    child.children.reverse()
                    last_else = child
                    index = base.children.index(child)

                elif isinstance(child, For_loopAST):
                    if child.parent is None:
                        child.parent = base
                    child.children = base.children[index - 4 : index]
                    base.children[index - 4 : index] = []
                    child.children.reverse()
                    # assign initialization
                    child.initialization = child.children[0]
                    child.initialization.parent = child
                    # assign condition
                    child.condition = child.children[1]
                    child.condition.parent = child
                    # assign increment
                    child.incr = child.children[2]
                    child.incr.parent = child
                    child.children = child.children[3:]
                    child.condition.in_loop = True
                    for ch in child.children:
                        ch.in_loop = True
                    index = base.children.index(child)

                elif isinstance(child, Scope_AST) or isinstance(child, FuncScopeAST):
                    # Parent of scope is base itself, if no parent is already found
                    # indexes["scope_depth"] += 1
                    if child.parent is None:
                        child.parent = base
                    new_index = self.searchPrevToken(
                        index=index, token="}", in_list=base.children
                    )
                    base.children[new_index : new_index + 1] = []
                    index = base.children.index(child)
                    child.children = base.children[new_index : index - 1]
                    child.children.reverse()
                    base.children[new_index:index] = []
                    index = base.children.index(child)
                    # indexes["last_scope"][(indexes["scope_depth"]-1)] += 1
                    if not (
                        isinstance(child.parent, If_CondAST)
                        or isinstance(child, FuncScopeAST)
                    ):
                        for ch in child.children:
                            ch.in_loop = True
                    indexes["last_instr"] = (
                        self.lastInstruction(index, base.children) + 1
                    )

                elif isinstance(child, DeclrAST):
                    last_decl = self.lastDeclaration(index, base.children)
                    # last_decl += 1
                    child.children = base.children[last_decl + 1 : index]
                    child.children.reverse()
                    base.children[last_decl + 1 : index] = []
                    index = base.children.index(child)
                    update_index = last_decl
                    update_index += 1

                elif isinstance(child, AssignAST):
                    child.children = base.children[index - 2 : index]
                    child.children.reverse()
                    base.children[index - 2 : index] = []
                    # # Add first child to symbol table if it isn't already in
                    # if not temp_symbol.exists(child.children[0].key):
                    #     # make one
                    #     new_object = child.children[0]
                    #     temp_symbol.insert(SymbolEntry(new_object))
                    # else:
                    #     raise AttributeError(f"Redeclaration of variable {child.children[0].key}")
                    index -= 2
                elif isinstance(child, VarDeclrAST):
                    if child.root.key == "assign":
                        child.children = base.children[index - 2 : index]
                        child.children.reverse()
                        base.children[index - 2 : index] = []
                        index -= 2
                    else:
                        child.children = base.children[index - 1 : index]
                        child.children.reverse()
                        base.children[index - 1 : index] = []
                        index -= 1
                elif isinstance(child, PrintfAST):
                    if child.root.value is None:
                        if len(child.children) > 0:
                            continue
                        child.args = base.children[index - len(child.args) : index]
                        child.args.reverse()
                        child.children = base.children[index - len(child.args) : index]
                        child.children.reverse()
                        base.children[index - len(child.args) : index] = []
                        index -= len(child.args)
                elif child.root.key == "deref":
                    child.children = base.children[index - 1 : index]
                    base.children[index - 1 : index] = []
                    index -= 1
                # connect children to this node
                child = base.children[index]
                for n in child.children:
                    n.parent = child

                    if (
                        isinstance(child, AST)
                        and child.symbolTable is not None
                        and child.symbolTable.parent is None
                        and n.symbolTable is not None
                    ):
                        # add child symbol table as parent symbol table of n symbol table
                        if isinstance(n, AST):
                            n.symbolTable.parent = child.symbolTable
                    if child.root.key == "declr" and child.root.value is not None:
                        if isinstance(n, AST):
                            n.root.value = child.root.value
                        elif isinstance(n, VarNode):
                            n.type = child.root.value

            elif isinstance(child, Node):
                if child.key == "}":
                    indexes["scope_depth"] += 1
                    # indexes["last_scope_open"] = index
                    # base.children[index:index+1] = []
                    # index -= 1

                if child.key == "{":
                    indexes["scope_depth"] -= 1
                    # base.children[index:index + 1] = []
                    # index -= 1

                if child.key == "term" and child.value is None:
                    child.value = base.children[index - 1].value

                if isinstance(base.children[index], AST):
                    child = base.children[index]
                    # connect children to this node
                    for n in child.children:
                        n.parent = child
                        if child.root.key == "declr" and child.root.value is not None:
                            if isinstance(n, AST):
                                n.root.value = child.root.value
                            elif isinstance(n, VarNode):
                                n.type = child.root.value

                elif isinstance(child, FuncParameter):
                    if not isinstance(
                        base.children[index - 1], FuncParameter
                    ) and isinstance(base.children[index - 1], Node):
                        child.value = base.children[index - 1].value
                        base.children[index - 1 : index] = []
                        index = base.children.index(child)

            index += 1
        base.children.reverse()
        return base

    def DFS(self, visited, ctx, root_name: str = "math"):
        if visited is None:
            visited = []
        s = list()
        a = AST(root=Node(root_name, None))
        s.append(ctx)
        # while there are still nodes to visit in the tree
        while len(s) > 0:
            v = s.pop()
            if v not in visited:
                visited.append(v)
                s.append(v)
                if isinstance(v, antlr4.tree.Tree.TerminalNodeImpl):
                    continue
                for child in v.getChildren():
                    s.append(child)
            else:
                v = self.visit_child(v)
                if v is None:
                    continue
                a.add_child(v)
        a = self.resolveTree(a)
        return a

    def resolve(
        self,
        ast_in: AST,
        in_loop: bool = False,
        in_func: bool = False,
        in_cond: bool = False,
    ):
        visited = list()
        not_visited = list()
        not_visited.append(ast_in)
        while len(not_visited) > 0:
            temp = not_visited.pop()
            if temp not in visited or isinstance(temp, CondAST):
                # if a scope, skip
                if not (isinstance(temp, Scope_AST) or isinstance(temp, FuncScopeAST)):
                    visited.append(temp)
                if isinstance(temp, Node):
                    continue
                if (
                    not (isinstance(temp, Scope_AST) or isinstance(temp, FuncScopeAST))
                    or temp is ast_in
                    or isinstance(temp.parent, Scope_AST)
                    or isinstance(temp.parent, FuncScopeAST)
                ):
                    # if isinstance(temp, For_loopAST):
                    #     if temp.initialization is not None:
                    #         not_visited.append(temp.initialization)
                    # if temp.incr is not None:
                    #     not_visited.append(temp.incr)
                    if (
                        isinstance(temp, While_loopAST)
                        or isinstance(temp, If_CondAST)
                        or isinstance(temp, For_loopAST)
                    ):
                        # not_visited.append(temp.condition)
                        visited.append(temp)
                        continue
                    for i in temp.children:
                        if isinstance(i, AST):
                            not_visited.append(i)
                else:
                    if (
                        isinstance(temp, For_loopAST)
                        and temp.initialization is not None
                    ):
                        not_visited.append(temp.initialization)
                    # if temp.condition is not None and not isinstance(temp.condition, Node):
                    #     not_visited.append(temp.condition)
        visited.reverse()
        self.handle(visited, in_loop, in_func, in_cond)
        return ast_in

    def handle(
        self,
        list_ast: list,
        in_loop: bool = False,
        in_func: bool = False,
        in_cond: bool = False,
    ):
        # TODO: handle function call
        # initialize queues
        updates_queue = []
        incr_queue = []
        decr_queue = []
        # flags
        evaluate = True
        temp_symbol = None
        nodes = []
        for ast in list_ast:
            if isinstance(ast, CommentAST):
                continue
            temp_parent = ast.parent
            symbol_table = ast.symbolTable
            while symbol_table is None and temp_parent is not None:
                symbol_table = temp_parent.symbolTable
                temp_parent = temp_parent.parent
            if symbol_table is None:
                raise RuntimeError("No symbol table found")
            temp_symbol = symbol_table
            if isinstance(ast, FuncCallAST):
                # check whether function is in symbol table
                temp_parent = ast
                temp_symbol = ast.symbolTable
                while temp_parent.parent is not None and temp_symbol is None:
                    temp_symbol = temp_parent.symbolTable
                    temp_parent = temp_parent.parent
                if temp_parent.symbolTable is None:
                    raise RuntimeError("Symbol table not found")
                while not temp_symbol.exists(ast.root.key):
                    temp_symbol = temp_symbol.parent
                    if temp_symbol.parent is None:
                        break
                if not temp_symbol.exists(ast.root.key):
                    raise AttributeError(f"Function {ast.root.key} not found in scope")
                # replace args
                for i in range(len(ast.args)):
                    if isinstance(ast.args[i], Node) and ast.args[i].key != "var":
                        continue
                    if not isinstance(ast.args[i], AST):
                        match = AST.getEntry(ast.args[i])
                        if isinstance(ast.args[i], Node):
                            match[0].addr = ast.args[i].addr
                        if match is None:
                            raise AttributeError(
                                f"Variable {ast.args[i].value} not found in scope"
                            )
                        if match[0] is None:
                            raise AttributeError(
                                f"Variable {ast.args[i].value} not found in scope"
                            )
                        ast.args[i] = match[0]
                if temp_parent.symbolTable is None:
                    raise RuntimeError("Symbol table not found")
                match_found = True
                for entry in temp_symbol.table:
                    # name match
                    if entry.name == ast.root.key:
                        match = entry
                        if len(entry.parameters) != len(ast.args):
                            match_found = False
                            if len(entry.parameters) > len(ast.args):
                                error = f"Too few arguments passed to function {entry.name}, expected {len(entry.parameters)}, got {len(ast.args)}\n"
                            else:
                                error = f"Too many arguments passed to function {entry.name}, expected {len(entry.parameters)}, got {len(ast.args)}\n"
                            # get instruction in the file where the warning is by using column and line number
                            f = open(self.file_name)
                            lines = f.readlines()
                            line = lines[ast.line - 1]
                            f.close()
                            # insert squiggly line
                            line = line[: ast.column] + "\u0332" + line[ast.column :]
                            line = "\033[95mError:\033[0m" + line.replace("\t", " ")
                            raise AttributeError(error + line)
                        for i in range(len(entry.parameters)):
                            current_param = entry.parameters[i]
                            current_arg = ast.args[i]
                            if isinstance(current_arg, AST):
                                continue
                            if isinstance(current_arg, VarNode):
                                # check all attributes
                                # if current_param.name != current_arg.key:
                                #     match_found = False
                                #     break
                                if current_param.const != current_arg.const:
                                    match_found = False
                                    break
                                if current_arg.addr:
                                    if current_param.ptr_level != 1:
                                        match_found = False
                                        break
                                else:
                                    if current_param.ptr != current_arg.ptr:
                                        match_found = False
                                        break
                                    if current_param.ptr_level != (
                                        current_arg.total_deref
                                        - current_arg.deref_level
                                    ):
                                        match_found = False
                                        break
                                if current_param.array != current_arg.array:
                                    match_found = False
                                    break
                                if current_param.type != current_arg.type:
                                    if (
                                        current_param.type,
                                        current_arg.type,
                                    ) not in conversions:
                                        match_found = False
                                        break
                                    if (
                                        current_arg.type,
                                        current_param.type,
                                    ) not in conv_promotions:
                                        self.warnings.append(
                                            f"Conversion from {current_arg.type} to {current_param.type} may cause loss of data"
                                        )
                            else:
                                if current_param.type != getType(current_arg.value):
                                    if (
                                        getType(current_arg.value),
                                        current_param.type,
                                    ) not in conversions:
                                        match_found = False
                                        break
                if not match_found:
                    raise AttributeError(f"Function {ast.root.key} not found")
                ast.root.type = match.type
                continue

            if isinstance(ast, FuncDeclAST):
                # add parent symbol table as parent of current symbol table
                ast.symbolTable.parent = ast.parent.symbolTable
                # check function was previously declared
                if ast.parent.symbolTable.exists(ast.root):
                    # check what is redeclared
                    match = ast.parent.symbolTable.lookup(ast.root)[0]
                    if match.type != ast.type:
                        raise AttributeError(
                            f"Redeclaration of function {ast.root.key} with different return type"
                        )
                    if len(match.parameters) != len(ast.params):
                        raise AttributeError(
                            f"Redeclaration of function {ast.root.key} with different number of parameters"
                        )
                    for param in ast.params:
                        if param not in match.parameters:
                            raise AttributeError(
                                f"Redeclaration of function {ast.root.key} with different parameters"
                            )
                    raise AttributeError(f"Redeclaration of function {ast.root.key}")
                new_entry = FuncSymbolEntry(ast.root)
                param_names = []
                for param in ast.params:
                    if param.key in param_names:
                        raise AttributeError(
                            f"Redefinition of parameter {param.key}"
                        )
                    new_entry.parameters.append(FunctionParameter(param))
                    ast.symbolTable.insert(SymbolEntry(param))
                    param_names.append(param.key)
                # add symbol table of ast to function entry in symbol table
                new_entry.symbol_table = ast.symbolTable
                # insert function into symbol table
                ast.parent.symbolTable.insert(new_entry)
                ast.symbolTable.parent = ast.parent.symbolTable
                node = ast
                continue
            if isinstance(ast, FuncScopeAST) or isinstance(ast, FuncDefnAST):
                # add parent symbol table as parent of current symbol table
                ast.symbolTable.parent = ast.parent.symbolTable
                temp_exists = False
                match = None
                if ast.parent.symbolTable.exists(ast.root):
                    temp_exists = True
                    matches = ast.parent.symbolTable.lookup(ast.root)
                    if len(matches) == 1:
                        if matches[0].defined:
                            raise AttributeError(
                                f"Redefinition of function {ast.root.key}"
                            )
                        match = matches[0]
                if not temp_exists:
                    new_entry = FuncSymbolEntry(ast.root)
                    param_names = []
                    for param in ast.params:
                        if param.key in param_names:
                            raise AttributeError(
                                f"Redefinition of parameter {param.key}"
                            )
                        new_entry.parameters.append(FunctionParameter(param))
                        ast.symbolTable.insert(SymbolEntry(param))
                        param_names.append(param.key)
                    new_entry.defined = True
                    # add symbol table of ast to function entry in symbol table
                    new_entry.symbol_table = ast.symbolTable
                    # insert function into symbol table
                    entry = ast.parent.symbolTable.insert(new_entry)
                    entry.symbol_table = ast.symbolTable
                else:
                    new_entry = FuncSymbolEntry(ast.root)
                    for param in ast.params:
                        new_entry.parameters.append(FunctionParameter(param))
                        ast.symbolTable.insert(SymbolEntry(param))
                    if new_entry != match:
                        if new_entry.type != match.type:
                            raise AttributeError(
                                f"Redefinition of function {ast.root.key} with different return type"
                            )
                        if len(new_entry.parameters) != len(match.parameters):
                            raise AttributeError(
                                f"Redefinition of function {ast.root.key} with different number of parameters"
                            )
                        for param in new_entry.parameters:
                            if param not in match.parameters:
                                raise AttributeError(
                                    f"Redefinition of function {ast.root.key} with different parameters"
                                )
                        raise AttributeError(
                            f"Redeclaration of function {ast.root.key} with different signature"
                        )
                    if match.defined:
                        raise AttributeError(f"Redefinition of function {ast.root.key}")
                    match.defined = True
                    # ast.parent.symbolTable.refresh()

                    # check if entries match
                # declare each parameter in your scope
                # handle what's in the function scope
                ast.children[-1].symbolTable = ast.symbolTable
                self.resolve(ast.children[-1], in_func=True)
                # update symbol table of function definition
                # if symbol_table is None:
                #     symbol_table = SymbolTable(ast.children[-1])
                #     ast.children[-1].symbolTable = symbol_table
                # symbol_table.parent = ast.symbolTable.parent
                # ast.symbolTable = symbol_table
                # symbol_table.owner = ast
                # link symbol table to function entry in global symbol table
                ast.symbolTable.parent.lookup(ast.root)[
                    0
                ].symbol_table = ast.symbolTable
                # print symbol table
                # print(f"Symbol table for {ast.root.key}:")
                # symbol_table.print()
                # functions
            if isinstance(ast, IncludeAST):
                # declare printf and scanf on the symbol table
                # printf
                printf = FuncSymbolEntry(VarNode("printf", None, "int"))
                printf.parameters.append(
                    FunctionParameter(FuncParameter("format", None, "string"))
                )
                printf.parameters.append(
                    FunctionParameter(FuncParameter("...", None, "void"))
                )
                printf.defined = True
                ast.parent.symbolTable.insert(printf)
                # scanf
                scanf = FuncSymbolEntry(VarNode("scanf", None, "int"))
                scanf.parameters.append(
                    FunctionParameter(FuncParameter("format", None, "string"))
                )
                scanf.parameters.append(
                    FunctionParameter(FuncParameter("...", None, "void"))
                )
                scanf.defined = True
                ast.parent.symbolTable.insert(scanf)

            if isinstance(ast, ArrayDeclAST):
                if temp_symbol is not None:
                    exists_state = temp_symbol.exists(ast.root.key[:-2])
                elif symbol_table is not None:
                    exists_state = symbol_table.exists(ast.root.key[:-2])
                else:
                    exists_state = False
                if exists_state:
                    match = temp_symbol.lookup(ast.root.key[:-2])[0]
                    if match.type != ast.type:
                        raise AttributeError(
                            f"Redeclaration of variable {match.name} with different type"
                        )
                    if not match.array:
                        raise AttributeError(
                            f"Redeclaration of variable {match.name} as array"
                        )
                else:
                    if ast.size < 0:
                        raise AttributeError(f"Array {ast.root.key} has invalid size")
                    # declare the array
                    new_array = ArrayNode(
                        ast.root.key[:-2],
                        None,
                        ast.type,
                        in_size=ast.size,
                        in_values=ast.values,
                        const=ast.root.const,
                        ptr=ast.root.ptr,
                        deref_level=ast.root.deref_level,
                        total_deref=ast.root.total_deref,
                        is_array=True,
                        const_ptr=ast.root.const and ast.root.ptr,
                    )
                    new_array.parent = ast.parent
                    new_array.values = ast.values
                    if len(new_array.values) > new_array.size:
                        raise AttributeError(
                            f"Too many values in array {new_array.key}"
                        )
                    if len(new_array.values) < new_array.size:
                        # fill the missing values with zeros
                        for i in range(new_array.size - len(new_array.values)):
                            new_array.values.append(
                                Node(ast.type, self.convert(0, new_array.type))
                            )
                    for value in new_array.values:
                        if isinstance(value, Node):
                            value.parent = new_array
                            if value.key != new_array.type:
                                self.warnings.append(
                                    f"Implicit conversion from {value.key} to {new_array.type} in "
                                    f"array {new_array.key} for element {value.value}"
                                )
                                value.value = self.convert(value.value, new_array.type)
                                value.key = new_array.type
                        if value is None:
                            # initialize the value to zero with the correct type
                            new_array.values[new_array.values.index(value)] = (
                                self.convert(0, new_array.type)
                            )
                    temp_symbol.insert(SymbolEntry(new_array))
                    temp_symbol.refresh()
                    node = new_array
                    ast.children = [node]

            if len(ast.children) == 0:
                continue
            if len(ast.children) > 0:
                handle = True
                temp_parent = ast.parent
                evaluate = True
                returned = False
                while temp_parent is not None:
                    if isinstance(temp_parent, FuncCallAST):
                        returned = True
                    if isinstance(temp_parent, While_loopAST) or isinstance(
                        temp_parent, ReturnInstr
                    ):
                        evaluate = False
                        break
                    temp_parent = temp_parent.parent
                for child in ast.children:
                    # unhandled trees
                    # check parent line:
                    # if it is referenced from a function parameter, then evaluate is false
                    temp_parent = child.parent
                    while temp_parent is not None:
                        if isinstance(temp_parent, FuncParameter):
                            evaluate = False
                            break
                        temp_parent = temp_parent.parent
                    if isinstance(ast, For_loopAST):
                        break
                    if (
                        isinstance(child, AST)
                        and not isinstance(ast, Scope_AST)
                        and not isinstance(ast, SwitchAST)
                        and not isinstance(child, FuncCallAST)
                    ):
                        handle = False
                        continue
                    if isinstance(ast, ArrayElementAST):
                        # check if the array was previously declared
                        while not temp_symbol.exists(ast.root.key):
                            temp_symbol = temp_symbol.parent
                            if temp_symbol is None:
                                # get instruction in the file where the warning is by using column and line number
                                f = open(self.file_name)
                                lines = f.readlines()
                                line = lines[ast.line - 1]
                                f.close()
                                # insert squiggly line
                                line = (
                                    line[: ast.column] + "\u0332" + line[ast.column :]
                                )
                                line = "\033[95mError:\033[0m" + line.replace("\t", " ")
                                raise AttributeError(
                                    f"Error at line {ast.line}:{ast.column}: Array {ast.root.key} was not declared\n"
                                    f"{line}"
                                )
                        match = temp_symbol.lookup(ast.root.key)[0]
                        ast.root.type = match.type
                        ast.children[0].type = match.type
                    # un-replaced rvars
                    if isinstance(child, Node) and child.key == "var":
                        # temp_parent = child.parent
                        # temp_symbol = child.parent.symbolTable
                        # symbol_table = child.parent.symbolTable
                        # search in symbol table
                        if temp_symbol is not None:
                            exists_state = temp_symbol.exists(child.value)
                        elif symbol_table is not None:
                            exists_state = symbol_table.exists(child.value)
                        else:
                            exists_state = False
                        temp_ast = ast
                        # search in parent scopes if not found
                        while (
                            not exists_state
                            and temp_ast is not None
                            and temp_ast.parent is not None
                            and temp_symbol.parent is not None
                        ):
                            temp_symbol = temp_symbol.parent
                            temp_ast = temp_ast.parent
                            if temp_symbol is not None:
                                exists_state = temp_symbol.exists(child.value)
                        if temp_parent is not None:
                            if evaluate:
                                evaluate = not ast.in_loop
                        if not temp_symbol.exists(child.value):
                            raise ReferenceError(
                                f"Variable {child.value} was not declared in this scope"
                            )
                        index = ast.children.index(child)
                        matches = temp_symbol.lookup(child.value)
                        if len(matches) == 0:
                            raise ReferenceError(
                                f"Variable {ast.children[0].key} undeclared"
                            )
                        if len(matches) > 1:
                            raise ReferenceError(
                                f"Multiple matches for variable {ast.children[0].key}"
                            )
                        matches[0].used = True
                        if not matches[0].returned:
                            matches[0].returned = returned
                            # matches[0].object.known = True
                        else:
                            matches[0].object.known = False
                            temp_parent = matches[0].object.parent
                            while temp_parent is not None and isinstance(
                                temp_parent, Node
                            ):
                                temp_parent.known = False
                                temp_symbol.update(temp_parent)
                                temp_symbol.refresh()
                                temp_parent = temp_parent.parent

                        if (
                            isinstance(matches[0].object, FuncParameter)
                            or child.type is None
                        ):
                            ast.children[index].type = matches[0].type
                        if evaluate and not in_loop and not matches[0].returned:
                            ast.children[index] = copy.deepcopy(matches[0].object)
                            if isinstance(ast.children[index], FuncParameter):
                                ast.children[index].parent = ast
                                evaluate = False

                if not handle:
                    continue
            if isinstance(ast, ReturnInstr):
                temp_parent = ast.parent
                while temp_parent is not None:
                    if isinstance(temp_parent, FuncDefnAST):
                        if temp_parent.type == "void" and ast.root.value != "void":
                            raise AttributeError(
                                f"void function '{temp_parent.root.key}' should not return a value"
                            )
                        if temp_parent.type != "void" and ast.root.value == "void":
                            raise AttributeError(
                                f"non-void function '{temp_parent.root.key}' should return a value"
                            )
                        break
                    temp_parent = temp_parent.parent
                continue
            if isinstance(ast, ScanfAST):
                for var in ast.variables:
                    match, total = AST.getEntry(var)
                    if total == -1:
                        raise ReferenceError(f"Variable {var.value} undeclared")
                    if total > 1:
                        raise ReferenceError(
                            f"Multiple matches for variable {var.value}"
                        )
                    ast.variables[ast.variables.index(var)] = match
                node = ast
            # conditional cases
            elif isinstance(ast, If_CondAST) or isinstance(ast, Else_CondAST):
                ast.symbolTable = temp_symbol
                self.resolve(ast.condition, in_cond=True, in_loop=in_loop)
                # handle for condition true
                self.resolve(ast.children[0], in_cond=True, in_loop=in_loop)
                self.resolve(ast.children[-1], in_cond=True, in_loop=in_loop)
                node = ast

            elif isinstance(ast, SwitchAST):
                # resolve the condition
                if not isinstance(ast.condition, Node):
                    self.resolve(ast.condition, in_cond=True, in_loop=in_loop)
                else:
                    # check if the condition is a variable
                    if ast.condition.key == "var":
                        while not temp_symbol.exists(ast.condition.value):
                            temp_symbol = temp_symbol.parent
                            if temp_symbol is None:
                                # get instruction in the file where the warning is by using column and line number
                                f = open(self.file_name)
                                lines = f.readlines()
                                line = lines[ast.line - 1]
                                f.close()
                                # insert squiggly line
                                line = (
                                    line[: ast.column] + "\u0332" + line[ast.column :]
                                )
                                line = "\033[95mError:\033[0m" + line.replace("\t", " ")
                                raise AttributeError(
                                    f"Error at line {ast.line}:{ast.column}: Array {ast.root.key} was not declared\n"
                                    f"{line}"
                                )
                        match = temp_symbol.lookup(ast.condition.value)
                        if len(match) == 0:
                            raise ReferenceError(
                                f"Variable {ast.condition.value} undeclared"
                            )
                        if not in_loop:
                            ast.condition = match[0].object
                        ast.condition.type = match[0].type
                # resolve the cases
                for case in ast.cases:
                    if isinstance(case.condition, AST):
                        case.condition.handle()
                    self.resolve(case, in_cond=True, in_loop=in_loop)
                self.resolve(ast.default, in_cond=True, in_loop=in_loop)
                # transform the switch into if-else
                # new_nodes = []
                # for i in range(len(ast.cases)):
                #     if not isinstance(ast.cases[i], DefaultAST):
                #         new_node = If_CondAST(Node(f"case_{i}", None), ast.cases[i].children)
                #     else:
                #         new_node = Else_CondAST(Node("default", None) ,ast.cases[i].children)
                #         new_nodes[-1].children.append(new_node)
                #     for child in new_node.children:
                #         child.parent = new_node
                #     new_node.parent = ast.parent if not isinstance(ast.cases[i], DefaultAST) else new_nodes[-1]
                #     new_node.in_loop = ast.in_loop
                #     new_node.in_func = ast.in_func
                #     new_node.column = ast.cases[i].column
                #     new_node.line = ast.cases[i].line
                #     new_node.symbolTable = ast.cases[i].symbolTable
                #     # create new condition: ast.condition == ast.cases[i].condition
                #     if not isinstance(ast.cases[i], DefaultAST):
                #         new_cond = TermAST(Node("cond", "=="))
                #         new_cond.parent = new_node
                #         new_cond.children.append(ast.condition)
                #         new_cond.children.append(ast.cases[i].condition[0])
                #         new_node.condition = new_cond
                #         for child in new_cond.children:
                #             child.parent = new_cond
                #     new_nodes.append(new_node)
                # index = ast.parent.children.index(ast)
                # ast.parent.children[index:index + 1] = new_nodes
                # for cond in new_nodes:
                #     if not in_loop and cond.condition is not None:
                #         cond.condition = cond.condition.handle()
                continue
            elif isinstance(ast, While_loopAST):
                self.resolve(ast.condition, in_loop=True, in_func=in_func)
                ast.symbolTable = temp_symbol
                self.resolve(ast.children[0], in_loop=True, in_func=in_func)
                node = ast

            elif isinstance(ast, For_loopAST):
                self.resolve(ast.initialization, in_loop=ast.in_loop, in_func=in_func)
                self.resolve(ast.condition, in_loop=True, in_func=in_func)
                self.resolve(ast.incr, in_loop=True, in_func=in_func)
                self.resolve(ast.children[0], in_loop=True, in_func=in_func)
                # check if increaser is declared
                while (
                    not temp_symbol.exists(ast.incr.children[0].value)
                    and temp_symbol.parent is not None
                ):
                    temp_symbol = temp_symbol.parent
                if temp_symbol.exists(ast.incr.children[0].value):
                    entry = temp_symbol.lookup(ast.incr.children[0].value)[0].object
                else:
                    raise ReferenceError(
                        f"Variable {ast.incr.children[0].key} was not declared"
                    )
                if entry is None:
                    if isinstance(ast.incr.children[0], VarNode):
                        raise ReferenceError(
                            f"Variable {ast.incr.children[0].key} was not declared"
                        )
                    raise ReferenceError("Incrementer must be a variable")
                # ast.incr.children[0] = entry
                # entry.parent = ast.incr
                ast.children[0].children.append(
                    InstrAST(Node("instr", None), [ast.incr])
                )
                ast.incr.parent = ast.children[0].children[-1]
                ast.children[0].children[-1].parent = ast.children[0]
                for child in ast.children[0].children:
                    child.parent = ast.children[0]
                # self.resolve(ast.initialization)
                temp = While_loopAST(Node("while", None), ast.children, ast.parent)
                temp.condition = ast.condition
                temp.symbolTable = ast.symbolTable
                temp.symbolTable.parent = ast.parent.symbolTable
                index = ast.parent.children.index(ast)
                ast.parent.children[index] = temp
                ast = temp
                for child in ast.children:
                    child.parent = ast
                ast.condition.parent = ast
                node = ast
            # Variable assignment handling
            elif ast.root.key == "assign" and ast.root.value is not None:
                if not isinstance(ast.children[0], VarNode):
                    raise AttributeError(
                        "Attempting to assign to a non variable type object"
                    )
                if not evaluate:
                    node = ast
                    continue
                # assign the value to the variable if it is not constant
                if not ast.children[0].const:
                    ast.children[0].value = ast.children[1].value
                    # get type
                    if isinstance(ast.children[1].value, int):
                        ast.children[0].type = "int"
                    elif isinstance(ast.children[1].value, float):
                        ast.children[0].type = "float"
                    elif (
                        isinstance(ast.children[1].value, str)
                        and len(ast.children[1].value) == 1
                    ):
                        ast.children[0].type = "char"
                    else:
                        raise TypeError(f"Wrong type assigned to {ast.children[0]}")
                    # Pointer depth check
                    if (
                        isinstance(ast.children[0], VarNode)
                        and isinstance(ast.children[1], VarNode)
                        and ast.children[0].ptr
                        and ast.children[1].ptr
                    ):
                        if (
                            ast.children[0].total_deref
                            != ast.children[1].total_deref + 1
                        ):
                            raise AttributeError(
                                f"Assignment of pointer {ast.children[1]} to {ast.children[0]} failed"
                            )
                    updates_queue.append(ast.children[0])
                    node = ast.children[0]
                    # refresh symbol table
                    # self.symbol_table.refresh()
                else:
                    raise AttributeError(
                        f"Attempting to modify a const variable {ast.children[0]}"
                    )
            # declaration handling
            elif isinstance(ast, InitAST):
                # go up one level in the symbol table tree
                temp_symbol = temp_symbol.parent
                temp_symbol.owner.symbolTable = temp_symbol
                # check if variable already exists
                if symbol_table.exists(ast.children[0]):
                    raise ReferenceError(
                        f"Redeclaration of variable {ast.children[0].key}"
                    )
                new_entry = ast.children[0]
                new_entry.type = ast.type
                new_entry.const = ast.const
                new_entry.value = ast.children[1].value
                temp_symbol.insert(SymbolEntry(new_entry))
                temp_symbol.refresh()
                updates_queue.append(new_entry)
                old_parent = ast.parent
                ast.parent = ast.parent.parent
                node = InstrAST(Node("instr", None), [new_entry])
                ast.parent.children.insert(ast.parent.children.index(old_parent), node)
                new_entry.parent = node
                node.parent = ast.parent
            elif isinstance(ast, DeclrAST):
                # if len(ast.children) != 1 or not isinstance(ast.children[0], VarNode):
                #     raise RuntimeError("Faulty declaration")
                nodes = []
                for i in range(len(ast.children)):
                    if symbol_table.exists(ast.children[i].key):
                        matches = temp_symbol.lookup(ast.children[i].key)
                        if len(matches) != 1:
                            raise ReferenceError(
                                f"Multiple matches for variable {ast.children[i].key}"
                            )
                        match = matches[0]
                        if match.initialized():
                            raise AttributeError(
                                f"Redeclaration of variable {ast.children[i].key}"
                            )
                        if not evaluate:
                            node = ast
                            continue
                    if (
                        ast.type != ast.children[i].type
                        and ast.children[i].value is not None
                        and not ast.children[i].cast
                    ):
                        if (ast.children[i].type, ast.type) not in conversions:
                            raise AttributeError("Variable assigned to wrong type")
                        if (ast.children[i].type, ast.type) not in conv_promotions:
                            # clang style warning (with warning in pink)
                            warning_str = "\033[95mwarning: \033[0m"
                            if (
                                ast.children[i].type == "float"
                                and ast.children[i].value is not None
                            ):
                                warning_str += f"implicit conversion from '{ast.children[i].type}' to '{ast.type}' changes value from {ast.children[i].value} to {self.convert(ast.children[i].value, ast.type)}"
                            if (
                                ast.children[i].type == "int"
                                and ast.children[i].value is not None
                            ):
                                warning_str += f"implicit conversion from '{ast.children[i].type}' to '{ast.type}' changes value from {ast.children[i].value} to {self.convert(ast.children[i].value, ast.type)}"
                            if (
                                ast.children[i].type == "char"
                                and ast.children[i].value is not None
                            ):
                                warning_str += f"implicit conversion from '{ast.children[i].type}' to '{ast.type}' changes value from {ast.children[i].value} to {self.convert(ast.children[i].value, ast.type)}"
                            # get instruction in the file where the warning is by using column and line number
                            f = open(self.file_name)
                            lines = f.readlines()
                            line = lines[ast.line - 1]
                            f.close()
                            # insert squiggly line
                            line = line[: ast.column] + "\u0332" + line[ast.column :]
                            warning_str += f"\n{ast.line}:{ast.column}: {line}"
                            self.warnings.append(warning_str)
                    node = ast.children[i]
                    node.type = ast.type
                    if node.value is not None and not isinstance(node.value, Node):
                        node.value = self.convert(node.value, ast.type)
                    # node.const = (ast.const is True)
                    if node.ptr and ast.const:
                        node.const_ptr = True
                    else:
                        node.const = ast.const is True
                    if not temp_symbol.exists(node):
                        if not evaluate:
                            node.value = None
                        temp_symbol.insert(SymbolEntry(node))
                    updates_queue.append(node)
                    nodes.append(node)

            elif isinstance(ast, AssignAST):
                # check if assign value is of a valid type
                if not (
                    isinstance(ast.children[1], Node)
                    or isinstance(ast.children[1], VarNode)
                    or isinstance(ast.children[1], ArrayNode)
                    or isinstance(ast.children[1], FuncCallAST)
                ):
                    raise RuntimeError(
                        f"'Invalid assignment for variable {ast.children[0].key}'"
                    )
                if isinstance(ast.children[1], VarNode):
                    rtype = ast.children[1].type
                elif isinstance(ast.children[1], Node):
                    rtype = ast.children[1].key
                elif isinstance(ast.children[1], ArrayNode):
                    rtype = ast.children[1].type
                elif isinstance(ast.children[1], FuncCallAST):
                    rtype = ast.children[1].root.type
                assignee = copy.copy(ast.children[0])
                if not evaluate:
                    node = ast
                    continue
                if ((
                    not isinstance(assignee, VarNode)
                    and not isinstance(assignee.parent, ArrayNode)
                ) and evaluate) or (
                    not evaluate
                    and not isinstance(assignee, Node)
                    and assignee.key != "var"
                ):
                    raise AttributeError("Attempting to assign to a non-variable type")
                if isinstance(assignee.parent, ArrayNode):
                    if assignee.parent.const:
                        raise AttributeError(
                            f"Attempting to modify a const array {assignee.parent.key}"
                        )
                    if rtype is None:
                        raise AttributeError(f"Type {rtype} does not exist")
                    if rtype != assignee.parent.type and not ast.children[1].cast:
                        if (assignee.parent.type, rtype) not in conversions:
                            raise AttributeError("Variable assigned to wrong type")
                        if (assignee.parent.type, rtype) not in conv_promotions:
                            self.warnings.append(
                                f"Implicit conversion from {ast.root.value} to {ast.children[0].type} for variable {ast.children[0].key}"
                            )
                else:
                    if assignee.const:
                        raise AttributeError(
                            f"Attempting to modify a const variable {assignee.key}"
                        )
                    if rtype is None:
                        raise AttributeError(f"Type {rtype} does not exist")
                    if (
                        isinstance(ast.children[1], FuncCallAST)
                        and rtype != assignee.type
                    ):
                        error = f"Variable assigned to function call of wrong type. Expected {assignee.type}, got {rtype}"
                        raise AttributeError(error)
                    if rtype != assignee.type and not ast.children[1].cast:
                        if (assignee.type, rtype) not in conversions:
                            raise AttributeError("Variable assigned to wrong type")
                        if (assignee.type, rtype) not in conv_promotions:
                            self.warnings.append(
                                f"Implicit conversion from {ast.root.value} to {ast.children[0].type} for variable {ast.children[0].key}"
                            )
                if (
                    isinstance(ast.children[0], VarNode)
                    and isinstance(ast.children[1], VarNode)
                    and ast.children[0].ptr
                    and ast.children[1].ptr
                    and ast.children[0].total_deref != ast.children[1].total_deref + 1
                ):
                    raise AttributeError(
                        f"Incompatible types for {ast.children[0].key} and {ast.children[1].key}."
                    )
                if isinstance(ast.children[0], VarNode) and not isinstance(
                    ast.children[1], VarNode
                ):
                    if ast.children[0].total_deref - ast.children[0].deref_level != 0:
                        raise AttributeError(
                            f"Assigning to pointer {ast.children[0].key} requires a pointer value."
                        )
                if not isinstance(assignee.parent, ArrayNode):
                    assignee.value = ast.children[1].value
                    if isinstance(assignee.parent, VarNode) and assignee.parent.ptr:
                        assignee.parent.value = assignee
                        temp_symbol.update(assignee.parent)
                else:
                    index = assignee.parent.values.index(assignee)
                    assignee.parent.values[index].value = ast.children[1].value
                    node = copy.deepcopy(assignee.parent.values[index])
                if isinstance(assignee, VarNode):
                    assignee.type = getType(assignee.value)
                    updates_queue.append(assignee)
                elif isinstance(assignee.parent, ArrayNode):
                    # directly update the array
                    # index = assignee.parent.values.index(assignee)
                    assignee.type = assignee.parent.type
                else:
                    updates_queue.append(assignee.parent)
                updates_queue.reverse()
                for instance in incr_queue:
                    instance = temp_symbol.lookup(instance)[0].object
                    instance.value += 1
                    temp_symbol.update(instance)
                for instance in decr_queue:
                    instance = temp_symbol.lookup(instance)[0].object
                    instance.value -= 1
                    temp_symbol.update(instance)
                for instance in updates_queue:
                    if not temp_symbol.exists(instance):
                        temp_symbol.insert(SymbolEntry(instance))
                    else:
                        temp_symbol.update(instance)
                temp_symbol.refresh()
                updates_queue.clear()
                incr_queue.clear()
                decr_queue.clear()
                if not isinstance(assignee.parent, ArrayNode):
                    node = assignee
            elif isinstance(ast, InstrAST):
                node = ast.handle()
                updates_queue.reverse()
                for instance in incr_queue:
                    # get the match from the nearest symbol table
                    if not isinstance(instance.parent, ArrayNode):
                        if isinstance(instance.parent, VarNode) and instance.parent.ptr:
                            temp_instance = copy.copy(instance)
                            temp_instance.value += 1
                            instance.parent.value = temp_instance
                            temp_symbol.update(instance.parent)
                            instance = copy.copy(temp_instance)
                            temp_symbol.update(instance)
                            temp_symbol.refresh()
                        else:
                            matches = temp_symbol.lookup(instance)
                            length = len(matches)
                            match = matches[0].object
                            if length == 0:
                                raise ReferenceError(
                                    f"Variable {instance.key} not found"
                                )
                            if length > 1:
                                raise ReferenceError(
                                    f"Multiple matches for variable {instance.key}"
                                )
                            instance = match
                            instance.value += 1
                            temp_symbol.update(instance)
                            temp_symbol.refresh()
                    else:
                        match = instance.parent
                        match.values[match.values.index(instance)].value += 1

                for instance in decr_queue:
                    # get the match from the nearest symbol table
                    if not isinstance(instance.parent, ArrayNode):
                        if isinstance(instance.parent, VarNode) and instance.parent.ptr:
                            temp_instance = copy.copy(instance)
                            temp_instance.value -= 1
                            instance.parent.value = temp_instance
                            temp_symbol.update(instance.parent)
                            instance = copy.copy(temp_instance)
                            temp_symbol.update(instance)
                            temp_symbol.refresh()
                        else:
                            matches = temp_symbol.lookup(instance)
                            length = len(matches)
                            match = matches[0].object
                            if length == 0:
                                raise ReferenceError(
                                    f"Variable {instance.key} not found"
                                )
                            if length > 1:
                                raise ReferenceError(
                                    f"Multiple matches for variable {instance.key}"
                                )
                            instance = match
                            instance.value -= 1
                            temp_symbol.update(instance)
                            temp_symbol.refresh()
                    else:
                        match = instance.parent
                        match.values[match.values.index(instance)].value -= 1

                for instance in updates_queue:
                    if not temp_symbol.exists(instance):
                        temp_symbol.insert(SymbolEntry(instance))
                    else:
                        temp_symbol.update(instance)

                temp_symbol.refresh()
                # temp_symbol = old_symbol
                updates_queue = []
                incr_queue = []
                decr_queue = []
            elif (
                isinstance(ast, TermAST) and ast.root.value in ["++", "--"] and evaluate
            ):
                node = ast
                if evaluate:
                    node = ast.children[0]
                    if isinstance(node.parent, ArrayNode):
                        node.value = (
                            node.value - 1 if ast.root.value == "--" else node.value + 1
                        )
                        updates_queue.append(node.parent)
                    elif isinstance(node, VarNode):
                        if ast.root.value == "++":
                            incr_queue.append(node)
                        elif ast.root.value == "--":
                            decr_queue.append(node)
                    else:
                        raise AttributeError(
                            f"Cannot increment or decrement non-variable {node.key}"
                        )
            elif isinstance(ast, CondAST):
                if evaluate and not ast.in_loop:
                    handle = True
                    for val in ast.children:
                        if (
                            (isinstance(val, Node)
                            and val.key == "var")
                            or isinstance(val, AST)
                        ):
                            node = ast
                            handle = False
                            break
                    if handle:
                        if isinstance(ast.children[0], Node):
                            ast.last_eval = ast.children[0].value
                        else:
                            ast.last_eval = copy.copy(ast).handle().value
            elif (
                isinstance(ast, TermAST)
                and ast.root.value in ["++", "--"]
                and not evaluate
            ) or (
                isinstance(ast, FactorAST)
                and ast.root.value in ["++", "--"]
                and not evaluate
            ) or (isinstance(ast, ArrayElementAST) and not evaluate):
                continue
            elif ast is not None:
                if isinstance(ast, TermAST) and ast.root.value == "++" and evaluate:
                    if isinstance(ast.parent, For_loopAST):
                        if ast.parent.incr == ast:
                            node = ast
                    else:
                        node = copy.copy(ast.children[0])
                        instance = copy.copy(node)
                        incr_queue.append(instance)
                elif isinstance(ast, TermAST) and ast.root.value == "--" and evaluate:
                    if isinstance(ast.parent, For_loopAST):
                        if ast.parent.incr == ast:
                            node = ast
                    else:
                        node = copy.copy(ast.children[0])
                        instance = copy.copy(node)
                        decr_queue.append(instance)
                elif (
                    isinstance(ast, TermAST)
                    and len(ast.children) > 1
                    and (
                        isinstance(ast.children[0], ArrayNode)
                        or isinstance(ast.children[1], ArrayNode)
                    )
                ):
                    # handle printf
                    node, warnings_handle = ast.handle()
                    for warning in warnings_handle:
                        # get line where warning is
                        warning_str = "\033[95mwarning: \033[0m"
                        f = open(self.file_name)
                        lines = f.readlines()
                        f.close()
                        line = lines[ast.line - 1]
                        # insert squiggly line
                        line = line[: ast.column] + "\u0332" + line[ast.column :]

                        warning_str += f"{warning}\n{ast.line}:{ast.column}: {line}"
                        self.warnings.append(warning_str)
                elif (
                    isinstance(ast, FactorAST)
                    and ast.root.value in ["++", "--"]
                    and evaluate
                ):
                    if isinstance(ast.parent, For_loopAST):
                        if ast.parent.incr == ast:
                            node = ast
                    else:
                        ast.children[0] = copy.deepcopy(ast.children[0])
                        node = ast.handle()
                        if ast.root.value == "++":
                            temp_symbol.update(node)
                            if isinstance(node.parent, VarNode) and node.parent.ptr:
                                node.parent.value = copy.deepcopy(node)
                                temp_symbol.update(node.parent)
                        if ast.root.value == "--":
                            temp_symbol.update(node)
                            if isinstance(node.parent, VarNode) and node.parent.ptr:
                                node.parent.value = copy.deepcopy(node)
                                temp_symbol.update(node.parent)
                elif isinstance(ast, PrintfAST):
                    # handle printf
                    node, warnings_handle = ast.handle()
                    for warning in warnings_handle:
                        # get line where warning is
                        warning_str = "\033[95mwarning: \033[0m"
                        f = open(self.file_name)
                        lines = f.readlines()
                        f.close()
                        line = lines[ast.line - 1]
                        # insert squiggly line
                        line = line[: ast.column] + "\u0332" + line[ast.column :]

                        warning_str += f"{warning}\n{ast.line}:{ast.column}: {line}"
                        self.warnings.append(warning_str)
                    # handle increment and decrement
                    for instance in incr_queue:
                        temp_instance = copy.copy(instance)
                        temp_instance.value += 1
                        temp_symbol.update(temp_instance)
                    for instance in decr_queue:
                        temp_instance = copy.copy(instance)
                        temp_instance.value -= 1
                        temp_symbol.update(temp_instance)
                    # update the symbol table
                    for instance in updates_queue:
                        temp_symbol.update(instance)
                    temp_symbol.refresh()
                    updates_queue = []
                    incr_queue = []
                    decr_queue = []

                else:
                    node = ast.handle()
            else:
                continue
            # Replace node
            if (
                not isinstance(ast, CondAST)
                and not isinstance(ast, InitAST)
                and not isinstance(ast, ArrayElementAST)
            ):
                if isinstance(ast.parent, For_loopAST):
                    if ast.parent.incr == ast:
                        ast.parent.incr = node
                else:
                    index = ast.parent.children.index(ast)
                    if len(nodes) == 0:
                        ast.parent.children[index] = node
                        if not (
                            isinstance(ast, DerefAST)
                            or (isinstance(ast.parent, VarNode) and ast.parent.ptr)
                            or isinstance(node.parent, ArrayNode)
                        ):
                            node.parent = ast.parent
                    else:
                        ast.parent.children = nodes
                        nodes = []
            else:
                if isinstance(ast, ArrayElementAST):
                    index = ast.parent.children.index(ast)
                    ast.parent.children[index] = node
                if isinstance(ast.parent, Else_CondAST):
                    ast.last_eval = not ast.last_eval
        if temp_symbol is not None:
            for instance in updates_queue:
                temp_symbol.update(instance)
            temp_symbol.refresh()
        symbol_table = temp_symbol
        return symbol_table

    def visitFile(self, ctx: FileParser.FileContext):
        """
        Math visit function
        :param ctx: context
        :return: AST
        """
        math_ast = self.DFS(None, ctx)
        math_ast.symbolTable = SymbolTable()
        math_ast.symbolTable.owner = math_ast
        return math_ast

    def visitInstr(self, ctx: FileParser.InstrContext):
        """
        Instruction visit
        :param ctx: context
        :return: AST
        """
        instr_ast = InstrAST()
        instr_ast.root = Node("instr", None)
        instr_ast.column = ctx.start.column
        instr_ast.line = ctx.start.line
        return instr_ast

    def visitExpr(self, ctx: FileParser.ExprContext):
        """
        Expression visit function
        :param ctx: context
        :return: AST
        """
        expr_ast = ExprAST()
        expr_ast.root = Node("expr", None)
        if len(ctx.children) == 3:
            expr_ast.root.value = ctx.children[1].getText()
        else:
            return None
        expr_ast.column = ctx.start.column
        expr_ast.line = ctx.start.line
        return expr_ast

    def visitPrintf(self, ctx: FileParser.PrintfContext):
        """
        Creates the node for printf function
        :param ctx: context
        :return: Node
        """

        out = PrintfAST(Node("printf", None))
        out.column = ctx.start.column
        out.line = ctx.start.line
        if ctx.print_val is not None:
            out.format_string = ctx.print_val.text[1:-1]  # printf
        if ctx.format_string is not None:
            out.format_string = ctx.format_string.text[1:-1]  # printf
        # split the format string into a list of strings and variables
        format_string = out.format_string
        format_string = format_string.replace("\\n", "\n")
        format_string = format_string.replace("\\t", "\t")
        format_string = format_string.replace("\\r", "\r")
        format_string = format_string.replace("\\v", "\v")
        format_string = format_string.replace("\\b", "\b")
        format_string = format_string.replace("\\a", "\a")
        format_string = format_string.replace("\\f", "\f")
        # format_string = format_string.replace("\\\\", "\\")
        # format_string = format_string.replace("\\\'", "\'")
        # format_string = format_string.replace("\\\"", "\"")
        # format_string = format_string.replace("\\\?", "\?")
        # format_string = format_string.replace("\\\0", "\0")
        format_string = format_string.replace(" ", "")
        out.format_specifiers += re.findall("%[0-9]*[discf]", format_string)
        # remove escape characters
        out.format_string = out.format_string.replace("\\n", "\\0A")
        out.format_string = out.format_string.replace("\\t", "\\09")
        out.format_string = out.format_string.replace("\\r", "\\0D")
        out.format_string = out.format_string.replace("\\v", "\\0B")
        out.format_string = out.format_string.replace("\\b", "\\08")
        out.format_string = out.format_string.replace("\\a", "\\07")
        out.format_string = out.format_string.replace("\\f", "\\0C")

        out.args = [None] * len(ctx.vars_)  # printf
        if len(out.args) != len(out.format_specifiers):
            raise AttributeError("Wrong number of arguments for printf")
        for i in range(len(ctx.vars_)):
            if isinstance(ctx.vars_[i].children[0], antlr4.tree.Tree.TerminalNodeImpl):
                new_node = Node("string", ctx.vars_[i].children[0].getText()[1:-1])
                out.args[i] = new_node
                out.children.append(new_node)
        return out

    def visitRvar(self, ctx: FileParser.RvarContext):
        """
        Right-hand side variable visit function
        :param ctx: context
        :return: Node
        """
        root = Node(keywords[0], ctx.children[0].getText())
        return root

    def visitRtype(self, ctx: FileParser.RtypeContext):
        """
        Right-hand side type visit function
        :param ctx: context
        :return: Node
        """
        if ctx.children[0].getText().isdigit():
            return Node(keywords_datatype[0], int(ctx.children[0].getText()))
        if isfloat(ctx.children[0].getText()):
            return Node(
                keywords_datatype[1], Decimal(ctx.children[0].getText()).__float__()
            )
        return Node(keywords_datatype[2], ctx.children[0].getText()[1:-1])

    def visitAssign(self, ctx: FileParser.AssignContext):
        """
        Assign operand visit function
        :param ctx: context
        :return: Node
        """
        root = AssignAST(Node("assign", None))
        root.column = ctx.start.column
        root.line = ctx.start.line
        return root

    def visitDeclr(self, ctx: FileParser.DeclrContext):
        """
        Declaration visit function
        :param ctx: context
        :return: AST
        """
        out = DeclrAST(Node("declr", None))
        out.column = ctx.start.column
        out.line = ctx.start.line
        index = 0
        if ctx.children[index].getText() == "const":
            out.const = True
            index += 1
        if ctx.children[index].getText() in keywords_datatype:
            out.type = ctx.children[index].getText()
        else:
            raise TypeError(
                f"Variable declared with invalid type {ctx.children[0].getText()}"
            )
        return out

    def visitVar_decl(self, ctx: FileParser.Var_declContext):
        """
        Variable declaration visit function
        :param ctx: context
        :return: VarNode || AST
        """
        if len(ctx.children) == 3:
            out = VarDeclrAST(Node("assign", None))
            out.column = ctx.start.column
            out.line = ctx.start.line
            return out
        out = VarDeclrAST(Node("var_declr", None))
        out.column = ctx.start.column
        out.line = ctx.start.line
        return out

    def visitLvar(self, ctx: FileParser.LvarContext):
        """
        Left hand side variable
        :param ctx: context
        :return: VarNode
        """
        if len(ctx.children) == 1:
            root = VarNode(ctx.children[-1].getText(), None, "")
            return root
        # If more than 1 element: it's a pointer
        ptr_len = len(ctx.ptr) if ctx.ptr is not None else 0
        is_ptr = ptr_len > 0
        root = VarNode(ctx.name.text, None, "", ptr=is_ptr, total_deref=ptr_len)
        return root

    def visitDeref(self, ctx: FileParser.DerefContext):
        """
        Dereference visit function
        :param ctx: context
        :return: VarNode
        """
        # STR rvar
        # STR deref
        out = DerefAST(Node("deref", None))
        out.column = ctx.start.column
        out.line = ctx.start.line
        return out

    def visitTerm(self, ctx: FileParser.TermContext):
        ast = TermAST()
        ast.column = ctx.start.column
        ast.line = ctx.start.line
        if len(ctx.children) == 3:
            ast.root = Node("term", ctx.children[1].getText())
        elif len(ctx.children) == 2:
            if ctx.children[0].getText() == "!":
                ast.root = Node("term", ctx.children[0].getText())
            else:
                ast.root = Node("term", ctx.children[1].getText())
        else:
            return None
        return ast

    def visitFactor(self, ctx: FileParser.FactorContext):
        ast = FactorAST()
        ast.column = ctx.start.column
        ast.line = ctx.start.line
        if len(ctx.children) == 2:
            ast.root = Node("factor", ctx.children[0].getText())
        else:
            return None
        return ast

    def visitPrimary(self, ctx: FileParser.PrimaryContext):
        ast = PrimaryAST()
        ast.column = ctx.start.column
        ast.line = ctx.start.line
        if len(ctx.children) == 2:
            ast.root = Node("primary", ctx.children[0].getText())
        else:
            return None
        return ast

    def visitScope(self, ctx: FileParser.ScopeContext):
        out = Scope_AST(Node("unnamed", None))
        out.column = ctx.start.column
        out.line = ctx.start.line
        return out

    def visitIf_cond(self, ctx: FileParser.If_condContext):
        out = If_CondAST(Node("If_cond", None))
        out.column = ctx.start.column
        out.line = ctx.start.line
        return out

    def visitElse_cond(self, ctx: FileParser.Else_condContext):
        out = Else_CondAST(Node("Else_cond", None))
        out.column = ctx.start.column
        out.line = ctx.start.line
        return out

    def visitWhile_loop(self, ctx: FileParser.While_loopContext):
        out = While_loopAST(Node("While_loop", None))
        out.column = ctx.start.column
        out.line = ctx.start.line
        return out

    def visitFor_loop(self, ctx: FileParser.For_loopContext):
        out = For_loopAST(Node("For_loop", None))
        out.column = ctx.start.column
        out.line = ctx.start.line
        return out

    def visitInit(self, ctx: FileParser.InitContext):
        if len(ctx.children) == 1:
            return None
        out = InitAST(Node("init", None))
        out.column = ctx.start.column
        out.line = ctx.start.line
        index = 0
        if ctx.children[index].getText() in keywords_datatype:
            out.type = ctx.children[index].getText()
        else:
            raise TypeError(
                f"Variable declared with invalid type {ctx.children[0].getText()}"
            )
        return out

    def visitCond(self, ctx: FileParser.CondContext):
        ast = CondAST()
        ast.column = ctx.start.column
        ast.line = ctx.start.line
        # cond : comp | expr
        if len(ctx.children) != 1:
            raise TypeError("Invalid condition")
        if isinstance(ctx.children[0], FileParser.CompContext):
            ast.root = Node("cond", None)
        elif isinstance(ctx.children[0], FileParser.ExprContext):
            ast.root = Node("expr", "const")
        return ast

    def visitIncr(self, ctx: FileParser.IncrContext):
        if isinstance(ctx.children[0], antlr4.tree.Tree.TerminalNodeImpl):
            # case for rvar INCR and rvar DECR
            out = TermAST(Node("factor", ctx.children[0].getText()))
        else:
            # case for INCR rvar and DECR rvar
            out = FactorAST(Node("term", ctx.children[1].getText()))
        out.column = ctx.start.column
        out.line = ctx.start.line
        return out

    def visitCont_instr(self, ctx: FileParser.Cont_instrContext):
        out = ContAST(Node("cont", None))
        out.column = ctx.start.column
        out.line = ctx.start.line
        return out

    def visitBreak_instr(self, ctx: FileParser.Break_instrContext):
        out = BreakAST(Node("break", None))
        out.column = ctx.start.column
        out.line = ctx.start.line
        return out

    def visitParam_list(self, ctx: FileParser.Param_listContext):
        out = FuncParametersAST(
            Node("parameter", None), parameters=[None for _ in ctx.params]
        )
        out.column = ctx.start.column
        out.line = ctx.start.line
        return out

    def visitParam_declr(self, ctx: FileParser.Param_declrContext):
        out = FuncParameter(
            key=ctx.var.text,
            value=None,
            vtype=ctx.type_.text,
            const=(ctx.const is not None),
            ptr=(ctx.ptr is not None and len(ctx.ptr) > 0),
            deref_level=0,
            total_deref=(len(ctx.ptr) if ctx.ptr is not None else 0),
            const_ptr=(ctx.const is not None and ctx.ptr is not None),
            reference=(ctx.reference is not None),
        )
        out.column = ctx.start.column
        out.line = ctx.start.line
        return out

    def visitFunc_defn(self, ctx: FileParser.Func_defnContext):
        out = FuncDefnAST(
            root=Node(ctx.name.text, None),
            const=(ctx.const is not None),
            return_type=ctx.type_.text,
            ptr=(len(ctx.ptr) > 0),
            ptr_level=(len(ctx.ptr)),
            symbolTable=SymbolTable(),
        )
        out.symbolTable.owner = out
        out.root = VarNode(
            out.root.key,
            out.root.value,
            out.type,
            out.const,
            out.ptr,
            total_deref=out.ptr_level,
            const_ptr=out.ptr and out.const,
        )
        out.column = ctx.start.column
        out.line = ctx.start.line
        return out

    def visitFunc_decl(self, ctx: FileParser.Func_declContext):
        out = FuncDeclAST(
            root=Node(ctx.name.text, None),
            const=(ctx.const is not None),
            return_type=ctx.type_.text,
            ptr=(len(ctx.ptr) > 0),
            ptr_level=(len(ctx.ptr)),
            symbolTable=SymbolTable(),
        )
        out.symbolTable.owner = out
        out.root = VarNode(
            out.root.key,
            out.root.value,
            out.type,
            out.const,
            out.ptr,
            total_deref=out.ptr_level,
            const_ptr=out.ptr and out.const,
        )
        out.column = ctx.start.column
        out.line = ctx.start.line
        return out

    def visitFunc_arg(self, ctx: FileParser.Func_argContext):
        return

    def visitArg_list(self, ctx: FileParser.Arg_listContext):
        """
        :return: Node with name args_list and value the number of arguments
        """
        # return Node("args_list", len(ctx.args))
        return

    def visitFunc_call(self, ctx: FileParser.Func_callContext):
        """
        :return: A FuncCallAST.
        Key is the name of the function being called and value is None.
        Args is an empty initialized list with the size of the number of arguments
        """
        out = FuncCallAST(Node(ctx.name.text, None))
        if ctx.args is not None:
            out.args = [None for _ in ctx.args.args]
        out.column = ctx.start.column
        out.line = ctx.start.line
        return out

    def visitFunc_scope(self, ctx: FileParser.Func_scopeContext):
        """
        :return: A FuncScopeAST.
        The key is the name of the function it belongs to.
        The value is None.
        """
        out = FuncScopeAST(
            Node(ctx.parentCtx.name.text, None), symbolTable=SymbolTable()
        )
        out.symbolTable.owner = out
        out.column = ctx.start.column
        out.line = ctx.start.line
        return out

    def visitReturn_instr(self, ctx: FileParser.Return_instrContext):
        out = ReturnInstr(Node("return", None))
        if ctx.ret_val is None:
            out.root.value = "void"
        out.column = ctx.start.column
        out.line = ctx.start.line
        return out

    def visitScanf(self, ctx: FileParser.ScanfContext):
        ast = ScanfAST(Node("scanf", None))
        ast.column = ctx.start.column
        ast.line = ctx.start.line
        # ast.variables = ctx.vars_
        ast.variables = [Node] * len(ctx.vars_)
        ast.format_string = ctx.format_string.text
        # split the format string into a list of strings and variables
        format_string = ast.format_string
        format_string = format_string.replace("\\n", "\n")
        format_string = format_string.replace("\\t", "\t")
        format_string = format_string.replace("\\r", "\r")
        format_string = format_string.replace("\\v", "\v")
        format_string = format_string.replace("\\b", "\b")
        format_string = format_string.replace("\\a", "\a")
        format_string = format_string.replace("\\f", "\f")
        format_string = format_string.replace("\\\\", "\\")
        format_string = format_string.replace("\\'", "'")
        format_string = format_string.replace('\\"', '"')
        format_string = format_string.replace("\\\?", "\?")
        format_string = format_string.replace("\\\0", "\0")
        format_string = format_string.replace(" ", "")
        ast.format_specifiers += re.findall(
            "%[0-9]*[disc]", format_string
        )  # find all format specifiers
        ast.variables = [
            self.visit_child(var.children[0]) for var in ctx.vars_
        ]  # scanf can have multiple variables
        for var in ast.variables:
            var.parent = ast
        return ast

    def visitArray_decl(self, ctx: FileParser.Array_declContext):
        ast = ArrayDeclAST(
            VarNode(
                ctx.name.text + "[]",
                None,
                ctx.type_.text,
                const=(ctx.const is not None),
                ptr=(len(ctx.ptr) > 0),
                deref_level=0,
                total_deref=(len(ctx.ptr) if ctx.ptr is not None else 0),
                const_ptr=(ctx.const is not None and len(ctx.ptr) > 0),
                is_array=True,
            ),
            arr_type=ctx.type_.text,
        )
        ast.column = ctx.start.column
        ast.line = ctx.start.line
        ast.root.parent = ast
        ast.values = [self.visit_child(value) for value in ctx.values]
        if ctx.size is not None:
            ast.size = int(ctx.size.text)
        else:
            ast.size = len(ast.values)
        if len(ast.values) > ast.size:
            raise RuntimeError(
                f"Too many values for array {ast.root.key} of size {ast.size} given in line "
                f"{ctx.start.line} column {ctx.start.column}"
            )
        # if len(ast.values) == 0:
        #     ast.values = [None for i in range(ast.size)]
        for value in ast.values:
            if value.key != ast.type:
                self.warnings.append(
                    f"Implicit cast from {value.key} to {ast.type} in line {ctx.start.line} for "
                    f"array element '{value.value}' of array '{ast.root.key}' with index "
                    f"{ast.values.index(value)}. This element will be casted to {self.convert(value.value, ast.type)}"
                )
            value.parent = ast
        return ast

    def visitArray_el(self, ctx: FileParser.Array_elContext):
        element = ArrayElementAST(Node("array_element", None))
        element.column = ctx.start.column
        element.line = ctx.start.line
        # get index
        if ctx.index is not None:
            element.root.value = int(ctx.index.text)
        return element

    def visitIncl_stat(self, ctx: FileParser.Incl_statContext):
        if ctx.library.text != "stdio":
            raise RuntimeError("Unsupported Library")
        out = IncludeAST(Node(f"{ctx.library.text}.h", None))
        out.column = ctx.start.column
        out.line = ctx.start.line
        return out

    def visitComp(self, ctx: FileParser.CompContext):
        # this is just an in-between node for an expression
        # check the operation
        if ctx.op.text in ["&&", "||"]:
            out = ExprAST(Node("expr", ctx.op.text))
        else:
            out = TermAST(Node("term", ctx.op.text))
        out.column = ctx.start.column
        out.line = ctx.start.line
        return out

    def visitSwitch_instr(self, ctx: FileParser.Switch_instrContext):
        out = SwitchAST(Node("switch", None))
        out.column = ctx.start.column
        out.line = ctx.start.line
        out.cases = [None] * len(ctx.case_list)
        out.has_default = ctx.default is not None
        return out

    def visitCase_instr(self, ctx: FileParser.Case_instrContext):
        out = CaseAST(Node("case", None))
        out.column = ctx.start.column
        out.line = ctx.start.line
        return out

    def visitDefault_instr(self, ctx: FileParser.Default_instrContext):
        out = DefaultAST(Node("default", None))
        out.column = ctx.start.column
        out.line = ctx.start.line
        return out

    def visitSwitch_scope(self, ctx: FileParser.Switch_scopeContext):
        out = SwitchScopeAST(Node("switch_scope", None))
        out.column = ctx.start.column
        out.line = ctx.start.line
        return out

    def visitComment(self, ctx: FileParser.CommentContext):
        out = CommentAST(Node("comment", ctx.com.text))
        if ctx.com.text.startswith("//"):
            out.root.value = "singleline"
            out.comment = ctx.com.text[2:]
        elif ctx.com.text.startswith("/*"):
            out.root.value = "multiline"
            out.comment = ctx.com.text[2:-2]
        # out.comment = ctx.com.text
        return out

    @staticmethod
    def convert(value, d_type):
        """
        help function for casting
        :param value: input_value
        :param d_type: cast type
        :return: cast value
        """
        try:
            if value is None:
                return value
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
                if value == 0:
                    # return utf-8 null character
                    return value
                return chr(value)
        except Exception:
            raise RuntimeError("Bad Cast")

    def warn(self):
        """
        print all warnings on console
        :return: None
        """
        for warn in self.warnings:
            print(warn)
