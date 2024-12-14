import enum
import subprocess

from .ast import AST, CommentAST, FuncDeclAST, FuncDefnAST, IncludeAST, InstrAST
from .node import Node
from .register_management import Registers


class ExecuteWith(enum.Enum):
    MARS = "mars"
    SPIM = "spim"


class Mips:
    """
    Converts AST of C language to MIPS
    """

    def __init__(self, input_ast, input_file: str = "out.asm"):
        self.ast = input_ast
        self.nodes: list[Node] = []
        self.file = input_file
        self.registers = Registers()

    def mips_dfs(self):
        # returns a list with all the AST in the tree in DFS order to convert to MIPS
        visited = []
        not_visited = [self.ast]
        while len(not_visited) > 0:
            temp = not_visited.pop()
            if temp not in visited:
                # if a scope, skip
                # if include instruction, skip
                if (
                    isinstance(temp, FuncDeclAST | FuncDefnAST | IncludeAST)
                    or (isinstance(temp, CommentAST) and temp.parent is self.ast)
                    or (isinstance(temp, InstrAST) and temp.parent is self.ast)
                ):
                    visited.append(temp)
                if isinstance(temp, AST):
                    for child in temp.children:
                        not_visited.append(child)
        visited.reverse()
        self.nodes = visited
        return visited

    def convert(self) -> None:
        """
        Converts AST to MIPS
        """
        self.mips_dfs()
        global_str = ""
        local_str = ""
        variables = ""
        with open(self.file, "w") as f:
            global_str, local_str, new_list = self.ast.mips(self.registers)
            for node in self.nodes:
                new_loc, new_glob, new_list = node.mips(self.registers)
                global_str += new_glob
                local_str += new_loc
            variables += ".data\n"
            for key, value in self.registers.globalObjects.data[0].items():
                variables += f'\t{value}: .asciiz "{key}"\n'
            for key, value in self.registers.globalObjects.data[1].items():
                variables += f"\t{value}: .float {key}\n"
            for key, value in self.registers.globalObjects.data[2].items():
                variables += f"\t{value}: .word {key}\n"
            for key, value in self.registers.globalObjects.data[3].items():
                variables += f"\t{value}: .byte {key}\n"
            for key, value in self.registers.globalObjects.data[5].items():
                variables += f"\t{value}: .space {key}\n"
            for key in self.registers.globalObjects.uninitialized[0]:  # char
                variables += f"\t.align 0\n\t{key}: .space 1\n"
            for key in self.registers.globalObjects.uninitialized[1]:  # float
                variables += f"\t.align 2\n\t{key}: .space 4\n"
            for key in self.registers.globalObjects.uninitialized[2]:  # int
                variables += f"\t.align 2\n\t{key}: .space 4\n"
            for key in self.registers.globalObjects.uninitialized[3]:  # array
                if key.type == "int":
                    variables += (
                        f"\t.align 2\n\tint_{key.key}: .space {(key.size + 1) * 4!s}\n"
                    )
                elif key.type == "float":
                    variables += (
                        f"\t.align 2\n\tflt_{key.key}: .space {(key.size + 1) * 4!s}\n"
                    )
                elif key.type == "char":
                    variables += (
                        f"\t.align 0\t\n\tchr_{key.key}: .space {key.size + 1!s}\n"
                    )
            variables += ".text\n"
            f.write(variables)
            f.write(global_str)
            f.write(local_str)
        print("MIPS code generated in " + self.file)

    def execute(
        self,
        execute_with: ExecuteWith = ExecuteWith.MARS,
        silent: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """
        Executes the MIPS code with Mars as default, but can be changed to SPIM
        :param execute_with: "mars" or "spim"
        :param disclaimer: If True, does not remove the disclaimer
        """
        if execute_with == ExecuteWith.MARS:
            output = subprocess.run(
                ["java", "-jar", "./Help/Mars4_5_Mod.jar", "nc", self.file],
                capture_output=True,
                text=True,
            )
            if not silent:
                print(output.stdout)
        elif execute_with == ExecuteWith.SPIM:
            output = subprocess.run(
                ["spim", "-file", self.file],
                capture_output=True,
                text=True,
            )
            output.stdout = output.stdout.replace(
                "SPIM Version 8.0 of January 8, 2010\nCopyright 1990-2010, James R. Larus.\n"
                "All Rights Reserved.\nSee the file README for a full copyright notice.\n"
                "Loaded: /usr/lib/spim/exceptions.s\n",
                "",
            )
            if not silent:
                print(output.stdout)
        return output

    @staticmethod
    def add(reg: str, op1: str, op2: str) -> str:
        return f"add {reg}, {op1}, {op2}"

    @staticmethod
    def addi(reg: str, op1: str, op2: str) -> str:
        return f"addi {reg}, {op1}, {op2}"

    @staticmethod
    def sub(reg: str, op1: str, op2: str) -> str:
        return f"sub {reg}, {op1}, {op2}"

    @staticmethod
    def mul(reg: str, op1: str, op2: str) -> str:
        return f"mul {reg}, {op1}, {op2}"

    @staticmethod
    def div(reg: str, op1: str, op2: str) -> str:
        return f"div {reg}, {op1}, {op2}"

    @staticmethod
    def and_(reg: str, op1: str, op2: str) -> str:
        return f"and {reg}, {op1}, {op2}"

    @staticmethod
    def or_(reg: str, op1: str, op2: str) -> str:
        return f"or {reg}, {op1}, {op2}"

    @staticmethod
    def xor(reg: str, op1: str, op2: str) -> str:
        return f"xor {reg}, {op1}, {op2}"

    @staticmethod
    def nor(reg: str, op1: str, op2: str) -> str:
        return f"nor {reg}, {op1}, {op2}"

    @staticmethod
    def slt(reg: str, op1: str, op2: str) -> str:
        return f"slt {reg}, {op1}, {op2}"

    @staticmethod
    def sll(reg: str, op1: str, op2: str) -> str:
        return f"sll {reg}, {op1}, {op2}"

    @staticmethod
    def srl(reg: str, op1: str, op2: str) -> str:
        return f"srl {reg}, {op1}, {op2}"

    @staticmethod
    def sra(reg: str, op1: str, op2: str) -> str:
        return f"sra {reg}, {op1}, {op2}"

    @staticmethod
    def lw(reg: str, op1: str, op2: str) -> str:
        return f"lw {reg}, {op1}({op2})"

    @staticmethod
    def sw(reg: str, op1: str, op2: str) -> str:
        return f"sw {reg}, {op1}({op2})"

    @staticmethod
    def beq(reg: str, op1: str, op2: str) -> str:
        return f"beq {reg}, {op1}, {op2}"

    @staticmethod
    def bne(reg: str, op1: str, op2: str) -> str:
        return f"bne {reg}, {op1}, {op2}"

    @staticmethod
    def j(op1: str) -> str:
        return f"j {op1}"

    @staticmethod
    def jr(reg: str) -> str:
        return f"jr {reg}"

    @staticmethod
    def jal(op1: str) -> str:
        return f"jal {op1}"

    @staticmethod
    def syscall() -> str:
        return "syscall"

    @staticmethod
    def label(label: str) -> str:
        return f"{label}:"

    @staticmethod
    def li(reg: str, op1: str) -> str:
        return f"li {reg}, {op1}"

    @staticmethod
    def la(reg: str, op1: str) -> str:
        return f"la {reg}, {op1}"

    @staticmethod
    def move(reg: str, op1: str) -> str:
        return f"move {reg}, {op1}"

    @staticmethod
    def mfhi(reg: str) -> str:
        return f"mfhi {reg}"

    @staticmethod
    def mflo(reg: str) -> str:
        return f"mflo {reg}"

    @staticmethod
    def nop() -> str:
        return "nop"

    @staticmethod
    def return_() -> str:
        return "jr $ra"

    @staticmethod
    def allocate_stack() -> str:
        # Allocate every register to stack
        out = "allocate_stack:\n"
        out += f"\t{Mips.addi(reg="$sp", op1="$sp", op2="-100")}\n"
        count = 0
        for i in range(2, 32):
            if i in [26, 27, 28, 29, 30]:
                count += 1
                continue
            out += f"\t{Mips.sw(reg=f"${i}", op1=str({(i - count)*4}), op2="$sp")}\n"
        # jump back to function
        out += f"\t{Mips.return_()}\n\n"
        return out

    @staticmethod
    def deallocate_stack() -> str:
        # Deallocate every register from stack
        out = "deallocate_stack:\n"
        count = 0
        for i in range(2, 32):
            if i in [26, 27, 28, 29, 30]:
                count += 1
                continue
            out += f"\t{Mips.lw(reg=f"${i}", op1=str({(i - count)*4}), op2="$sp")}\n"
        out += f"\t{Mips.addi(reg="$sp", op1="$sp", op2="100")}\n"
        # jump back to function
        out += f"\t{Mips.return_()}\n\n"
        return out
