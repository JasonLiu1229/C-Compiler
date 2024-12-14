# Generated from ./Grammars/File.g4 by ANTLR 4.13.1
from antlr4 import *
if "." in __name__:
    from .FileParser import FileParser
else:
    from FileParser import FileParser

# This class defines a complete generic visitor for a parse tree produced by FileParser.

class FileVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by FileParser#file.
    def visitFile(self, ctx:FileParser.FileContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#incl_stat.
    def visitIncl_stat(self, ctx:FileParser.Incl_statContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#instr.
    def visitInstr(self, ctx:FileParser.InstrContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#declr.
    def visitDeclr(self, ctx:FileParser.DeclrContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#printf.
    def visitPrintf(self, ctx:FileParser.PrintfContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#printf_arg.
    def visitPrintf_arg(self, ctx:FileParser.Printf_argContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#scanf.
    def visitScanf(self, ctx:FileParser.ScanfContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#scanf_arg.
    def visitScanf_arg(self, ctx:FileParser.Scanf_argContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#comment.
    def visitComment(self, ctx:FileParser.CommentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#enum_declr.
    def visitEnum_declr(self, ctx:FileParser.Enum_declrContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#func_defn.
    def visitFunc_defn(self, ctx:FileParser.Func_defnContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#func_decl.
    def visitFunc_decl(self, ctx:FileParser.Func_declContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#param_list.
    def visitParam_list(self, ctx:FileParser.Param_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#param_declr.
    def visitParam_declr(self, ctx:FileParser.Param_declrContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#func_call.
    def visitFunc_call(self, ctx:FileParser.Func_callContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#arg_list.
    def visitArg_list(self, ctx:FileParser.Arg_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#func_arg.
    def visitFunc_arg(self, ctx:FileParser.Func_argContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#func_scope.
    def visitFunc_scope(self, ctx:FileParser.Func_scopeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#scope.
    def visitScope(self, ctx:FileParser.ScopeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#return_instr.
    def visitReturn_instr(self, ctx:FileParser.Return_instrContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#switch_instr.
    def visitSwitch_instr(self, ctx:FileParser.Switch_instrContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#case_instr.
    def visitCase_instr(self, ctx:FileParser.Case_instrContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#default_instr.
    def visitDefault_instr(self, ctx:FileParser.Default_instrContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#switch_scope.
    def visitSwitch_scope(self, ctx:FileParser.Switch_scopeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#cont_instr.
    def visitCont_instr(self, ctx:FileParser.Cont_instrContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#break_instr.
    def visitBreak_instr(self, ctx:FileParser.Break_instrContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#if_cond.
    def visitIf_cond(self, ctx:FileParser.If_condContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#else_cond.
    def visitElse_cond(self, ctx:FileParser.Else_condContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#while_loop.
    def visitWhile_loop(self, ctx:FileParser.While_loopContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#for_loop.
    def visitFor_loop(self, ctx:FileParser.For_loopContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#init.
    def visitInit(self, ctx:FileParser.InitContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#array_decl.
    def visitArray_decl(self, ctx:FileParser.Array_declContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#cond.
    def visitCond(self, ctx:FileParser.CondContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#incr.
    def visitIncr(self, ctx:FileParser.IncrContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#var_decl.
    def visitVar_decl(self, ctx:FileParser.Var_declContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#assign.
    def visitAssign(self, ctx:FileParser.AssignContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#array_el.
    def visitArray_el(self, ctx:FileParser.Array_elContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#deref.
    def visitDeref(self, ctx:FileParser.DerefContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#lvar.
    def visitLvar(self, ctx:FileParser.LvarContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#rvar.
    def visitRvar(self, ctx:FileParser.RvarContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#comp.
    def visitComp(self, ctx:FileParser.CompContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#expr.
    def visitExpr(self, ctx:FileParser.ExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#term.
    def visitTerm(self, ctx:FileParser.TermContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#factor.
    def visitFactor(self, ctx:FileParser.FactorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#primary.
    def visitPrimary(self, ctx:FileParser.PrimaryContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by FileParser#rtype.
    def visitRtype(self, ctx:FileParser.RtypeContext):
        return self.visitChildren(ctx)



del FileParser