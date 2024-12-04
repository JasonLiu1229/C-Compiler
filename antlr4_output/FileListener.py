# Generated from ./Grammars/File.g4 by ANTLR 4.13.1
from antlr4 import *
if "." in __name__:
    from .FileParser import FileParser
else:
    from FileParser import FileParser

# This class defines a complete listener for a parse tree produced by FileParser.
class FileListener(ParseTreeListener):

    # Enter a parse tree produced by FileParser#file.
    def enterFile(self, ctx:FileParser.FileContext):
        pass

    # Exit a parse tree produced by FileParser#file.
    def exitFile(self, ctx:FileParser.FileContext):
        pass


    # Enter a parse tree produced by FileParser#incl_stat.
    def enterIncl_stat(self, ctx:FileParser.Incl_statContext):
        pass

    # Exit a parse tree produced by FileParser#incl_stat.
    def exitIncl_stat(self, ctx:FileParser.Incl_statContext):
        pass


    # Enter a parse tree produced by FileParser#instr.
    def enterInstr(self, ctx:FileParser.InstrContext):
        pass

    # Exit a parse tree produced by FileParser#instr.
    def exitInstr(self, ctx:FileParser.InstrContext):
        pass


    # Enter a parse tree produced by FileParser#declr.
    def enterDeclr(self, ctx:FileParser.DeclrContext):
        pass

    # Exit a parse tree produced by FileParser#declr.
    def exitDeclr(self, ctx:FileParser.DeclrContext):
        pass


    # Enter a parse tree produced by FileParser#printf.
    def enterPrintf(self, ctx:FileParser.PrintfContext):
        pass

    # Exit a parse tree produced by FileParser#printf.
    def exitPrintf(self, ctx:FileParser.PrintfContext):
        pass


    # Enter a parse tree produced by FileParser#printf_arg.
    def enterPrintf_arg(self, ctx:FileParser.Printf_argContext):
        pass

    # Exit a parse tree produced by FileParser#printf_arg.
    def exitPrintf_arg(self, ctx:FileParser.Printf_argContext):
        pass


    # Enter a parse tree produced by FileParser#scanf.
    def enterScanf(self, ctx:FileParser.ScanfContext):
        pass

    # Exit a parse tree produced by FileParser#scanf.
    def exitScanf(self, ctx:FileParser.ScanfContext):
        pass


    # Enter a parse tree produced by FileParser#scanf_arg.
    def enterScanf_arg(self, ctx:FileParser.Scanf_argContext):
        pass

    # Exit a parse tree produced by FileParser#scanf_arg.
    def exitScanf_arg(self, ctx:FileParser.Scanf_argContext):
        pass


    # Enter a parse tree produced by FileParser#comment.
    def enterComment(self, ctx:FileParser.CommentContext):
        pass

    # Exit a parse tree produced by FileParser#comment.
    def exitComment(self, ctx:FileParser.CommentContext):
        pass


    # Enter a parse tree produced by FileParser#enum_declr.
    def enterEnum_declr(self, ctx:FileParser.Enum_declrContext):
        pass

    # Exit a parse tree produced by FileParser#enum_declr.
    def exitEnum_declr(self, ctx:FileParser.Enum_declrContext):
        pass


    # Enter a parse tree produced by FileParser#func_defn.
    def enterFunc_defn(self, ctx:FileParser.Func_defnContext):
        pass

    # Exit a parse tree produced by FileParser#func_defn.
    def exitFunc_defn(self, ctx:FileParser.Func_defnContext):
        pass


    # Enter a parse tree produced by FileParser#func_decl.
    def enterFunc_decl(self, ctx:FileParser.Func_declContext):
        pass

    # Exit a parse tree produced by FileParser#func_decl.
    def exitFunc_decl(self, ctx:FileParser.Func_declContext):
        pass


    # Enter a parse tree produced by FileParser#param_list.
    def enterParam_list(self, ctx:FileParser.Param_listContext):
        pass

    # Exit a parse tree produced by FileParser#param_list.
    def exitParam_list(self, ctx:FileParser.Param_listContext):
        pass


    # Enter a parse tree produced by FileParser#param_declr.
    def enterParam_declr(self, ctx:FileParser.Param_declrContext):
        pass

    # Exit a parse tree produced by FileParser#param_declr.
    def exitParam_declr(self, ctx:FileParser.Param_declrContext):
        pass


    # Enter a parse tree produced by FileParser#func_call.
    def enterFunc_call(self, ctx:FileParser.Func_callContext):
        pass

    # Exit a parse tree produced by FileParser#func_call.
    def exitFunc_call(self, ctx:FileParser.Func_callContext):
        pass


    # Enter a parse tree produced by FileParser#arg_list.
    def enterArg_list(self, ctx:FileParser.Arg_listContext):
        pass

    # Exit a parse tree produced by FileParser#arg_list.
    def exitArg_list(self, ctx:FileParser.Arg_listContext):
        pass


    # Enter a parse tree produced by FileParser#func_arg.
    def enterFunc_arg(self, ctx:FileParser.Func_argContext):
        pass

    # Exit a parse tree produced by FileParser#func_arg.
    def exitFunc_arg(self, ctx:FileParser.Func_argContext):
        pass


    # Enter a parse tree produced by FileParser#func_scope.
    def enterFunc_scope(self, ctx:FileParser.Func_scopeContext):
        pass

    # Exit a parse tree produced by FileParser#func_scope.
    def exitFunc_scope(self, ctx:FileParser.Func_scopeContext):
        pass


    # Enter a parse tree produced by FileParser#scope.
    def enterScope(self, ctx:FileParser.ScopeContext):
        pass

    # Exit a parse tree produced by FileParser#scope.
    def exitScope(self, ctx:FileParser.ScopeContext):
        pass


    # Enter a parse tree produced by FileParser#return_instr.
    def enterReturn_instr(self, ctx:FileParser.Return_instrContext):
        pass

    # Exit a parse tree produced by FileParser#return_instr.
    def exitReturn_instr(self, ctx:FileParser.Return_instrContext):
        pass


    # Enter a parse tree produced by FileParser#switch_instr.
    def enterSwitch_instr(self, ctx:FileParser.Switch_instrContext):
        pass

    # Exit a parse tree produced by FileParser#switch_instr.
    def exitSwitch_instr(self, ctx:FileParser.Switch_instrContext):
        pass


    # Enter a parse tree produced by FileParser#case_instr.
    def enterCase_instr(self, ctx:FileParser.Case_instrContext):
        pass

    # Exit a parse tree produced by FileParser#case_instr.
    def exitCase_instr(self, ctx:FileParser.Case_instrContext):
        pass


    # Enter a parse tree produced by FileParser#default_instr.
    def enterDefault_instr(self, ctx:FileParser.Default_instrContext):
        pass

    # Exit a parse tree produced by FileParser#default_instr.
    def exitDefault_instr(self, ctx:FileParser.Default_instrContext):
        pass


    # Enter a parse tree produced by FileParser#switch_scope.
    def enterSwitch_scope(self, ctx:FileParser.Switch_scopeContext):
        pass

    # Exit a parse tree produced by FileParser#switch_scope.
    def exitSwitch_scope(self, ctx:FileParser.Switch_scopeContext):
        pass


    # Enter a parse tree produced by FileParser#cont_instr.
    def enterCont_instr(self, ctx:FileParser.Cont_instrContext):
        pass

    # Exit a parse tree produced by FileParser#cont_instr.
    def exitCont_instr(self, ctx:FileParser.Cont_instrContext):
        pass


    # Enter a parse tree produced by FileParser#break_instr.
    def enterBreak_instr(self, ctx:FileParser.Break_instrContext):
        pass

    # Exit a parse tree produced by FileParser#break_instr.
    def exitBreak_instr(self, ctx:FileParser.Break_instrContext):
        pass


    # Enter a parse tree produced by FileParser#if_cond.
    def enterIf_cond(self, ctx:FileParser.If_condContext):
        pass

    # Exit a parse tree produced by FileParser#if_cond.
    def exitIf_cond(self, ctx:FileParser.If_condContext):
        pass


    # Enter a parse tree produced by FileParser#else_cond.
    def enterElse_cond(self, ctx:FileParser.Else_condContext):
        pass

    # Exit a parse tree produced by FileParser#else_cond.
    def exitElse_cond(self, ctx:FileParser.Else_condContext):
        pass


    # Enter a parse tree produced by FileParser#while_loop.
    def enterWhile_loop(self, ctx:FileParser.While_loopContext):
        pass

    # Exit a parse tree produced by FileParser#while_loop.
    def exitWhile_loop(self, ctx:FileParser.While_loopContext):
        pass


    # Enter a parse tree produced by FileParser#for_loop.
    def enterFor_loop(self, ctx:FileParser.For_loopContext):
        pass

    # Exit a parse tree produced by FileParser#for_loop.
    def exitFor_loop(self, ctx:FileParser.For_loopContext):
        pass


    # Enter a parse tree produced by FileParser#init.
    def enterInit(self, ctx:FileParser.InitContext):
        pass

    # Exit a parse tree produced by FileParser#init.
    def exitInit(self, ctx:FileParser.InitContext):
        pass


    # Enter a parse tree produced by FileParser#array_decl.
    def enterArray_decl(self, ctx:FileParser.Array_declContext):
        pass

    # Exit a parse tree produced by FileParser#array_decl.
    def exitArray_decl(self, ctx:FileParser.Array_declContext):
        pass


    # Enter a parse tree produced by FileParser#cond.
    def enterCond(self, ctx:FileParser.CondContext):
        pass

    # Exit a parse tree produced by FileParser#cond.
    def exitCond(self, ctx:FileParser.CondContext):
        pass


    # Enter a parse tree produced by FileParser#incr.
    def enterIncr(self, ctx:FileParser.IncrContext):
        pass

    # Exit a parse tree produced by FileParser#incr.
    def exitIncr(self, ctx:FileParser.IncrContext):
        pass


    # Enter a parse tree produced by FileParser#var_decl.
    def enterVar_decl(self, ctx:FileParser.Var_declContext):
        pass

    # Exit a parse tree produced by FileParser#var_decl.
    def exitVar_decl(self, ctx:FileParser.Var_declContext):
        pass


    # Enter a parse tree produced by FileParser#assign.
    def enterAssign(self, ctx:FileParser.AssignContext):
        pass

    # Exit a parse tree produced by FileParser#assign.
    def exitAssign(self, ctx:FileParser.AssignContext):
        pass


    # Enter a parse tree produced by FileParser#array_el.
    def enterArray_el(self, ctx:FileParser.Array_elContext):
        pass

    # Exit a parse tree produced by FileParser#array_el.
    def exitArray_el(self, ctx:FileParser.Array_elContext):
        pass


    # Enter a parse tree produced by FileParser#deref.
    def enterDeref(self, ctx:FileParser.DerefContext):
        pass

    # Exit a parse tree produced by FileParser#deref.
    def exitDeref(self, ctx:FileParser.DerefContext):
        pass


    # Enter a parse tree produced by FileParser#lvar.
    def enterLvar(self, ctx:FileParser.LvarContext):
        pass

    # Exit a parse tree produced by FileParser#lvar.
    def exitLvar(self, ctx:FileParser.LvarContext):
        pass


    # Enter a parse tree produced by FileParser#rvar.
    def enterRvar(self, ctx:FileParser.RvarContext):
        pass

    # Exit a parse tree produced by FileParser#rvar.
    def exitRvar(self, ctx:FileParser.RvarContext):
        pass


    # Enter a parse tree produced by FileParser#comp.
    def enterComp(self, ctx:FileParser.CompContext):
        pass

    # Exit a parse tree produced by FileParser#comp.
    def exitComp(self, ctx:FileParser.CompContext):
        pass


    # Enter a parse tree produced by FileParser#expr.
    def enterExpr(self, ctx:FileParser.ExprContext):
        pass

    # Exit a parse tree produced by FileParser#expr.
    def exitExpr(self, ctx:FileParser.ExprContext):
        pass


    # Enter a parse tree produced by FileParser#term.
    def enterTerm(self, ctx:FileParser.TermContext):
        pass

    # Exit a parse tree produced by FileParser#term.
    def exitTerm(self, ctx:FileParser.TermContext):
        pass


    # Enter a parse tree produced by FileParser#factor.
    def enterFactor(self, ctx:FileParser.FactorContext):
        pass

    # Exit a parse tree produced by FileParser#factor.
    def exitFactor(self, ctx:FileParser.FactorContext):
        pass


    # Enter a parse tree produced by FileParser#primary.
    def enterPrimary(self, ctx:FileParser.PrimaryContext):
        pass

    # Exit a parse tree produced by FileParser#primary.
    def exitPrimary(self, ctx:FileParser.PrimaryContext):
        pass


    # Enter a parse tree produced by FileParser#rtype.
    def enterRtype(self, ctx:FileParser.RtypeContext):
        pass

    # Exit a parse tree produced by FileParser#rtype.
    def exitRtype(self, ctx:FileParser.RtypeContext):
        pass



del FileParser