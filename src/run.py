import argparse
import contextlib
import os
import subprocess

from antlr4 import CommonTokenStream, FileStream

from antlr4_output.FileLexer import FileLexer
from antlr4_output.FileParser import FileParser
from src.ast import AST

from .ast import ErrorListener
from .ast_creator import AstCreator
from .mips import ExecuteWith, Mips


class ExecutionError(Exception):
    pass


class CompilationError(Exception):
    pass


def run_file(
    directory: str,
    file_type: str,
    filename: str,
    verbose: bool = True,
    no_warning: bool = False,
    execute_with: ExecuteWith | None = None,
    disclaimer: bool = True,
    silent: bool = False,
    visualise: bool = False,
) -> subprocess.CompletedProcess[str] | None:
    try:
        ast, generator = generate_mips_code(
            directory, file_type, filename, verbose, no_warning, silent, visualise
        )
    except Exception as e:
        raise CompilationError("Error in generating MIPS code") from e

    try:
        if execute_with is not None:
            return generator.execute(execute_with, silent)
    except Exception as e:
        with contextlib.suppress(Exception):
            if verbose and not silent:
                ast.print(4, True, filename)
                if ast.symbolTable:
                    ast.symbolTable.print(True)
        raise ExecutionError() from e


def generate_mips_code(
    directory, file_type, filename, verbose, no_warning, silent, visualise
) -> tuple[AST, Mips]:
    file_path = os.path.join(directory, filename + file_type)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    input_stream = FileStream(file_path)
    # Create error listener
    error_listener = ErrorListener()
    # Create lexer and parser
    lexer = FileLexer(input_stream)
    parser = FileParser(CommonTokenStream(lexer))
    # Remove previous error listener and add new error listener to lexer
    lexer.removeErrorListeners()
    lexer.addErrorListener(error_listener)
    # Remove previous error listener and add new error listener to parser
    parser.removeErrorListeners()
    parser.addErrorListener(error_listener)
    # Parse the input stream
    parse_tree = parser.file_()
    # create ast
    visitor = AstCreator(filename=file_path)
    ast = visitor.visit(parse_tree)
    # handle tree
    ast = visitor.resolve(ast)
    # check if the main function exists
    if ast.symbolTable and not ast.symbolTable.exists("main"):
        raise Exception("No main function found")
        # delete unused variables and generate warning if there are any
    visitor.warnings += ast.delete_unused_variables(visitor.file_name)
    # print the ast if verbose is true
    if verbose and not silent:
        ast.print(4, True, filename)
        if ast.symbolTable:
            ast.symbolTable.print(True)
        # print warnings if there are any warnings and no_warning is false
    if not no_warning and not silent:
        visitor.warn()
        # create dot file
    # if visualise:
    #     ast.dot_language(filename, visitor.symbol_table)
    # generator = LLVM(ast,  "../Output/" + filename + ".ll")
    # generator.convert()
    # generator.execute()

    generator = Mips(ast, f"./MIPS_output/{filename}.asm")
    generator.convert()
    return ast, generator


def run(
    directory: str,
    file_type: str,
    filenames: list,
    verbose: bool = True,
    no_warning: bool = False,
    execute_with: ExecuteWith | None = None,
    disclaimer: bool = True,
    silent: bool = False,
    visualise: bool = False,
) -> subprocess.CompletedProcess[str] | None:
    for filename in filenames:
        try:
            return run_file(
                directory,
                file_type,
                filename,
                verbose,
                no_warning,
                execute_with,
                disclaimer,
                silent,
                visualise,
            )
        except ExecutionError as e:
            raise RuntimeError(f"Failed to execute {filename}") from e
    return None


def main():
    parser = argparse.ArgumentParser(
        prog="Compiler", description="Compiles the C files"
    )
    parser.add_argument(
        "-d",
        "--directory",
        help="directory of the file that needs to be parsed",
        required=True,
    )
    parser.add_argument(
        "-t", "--type", help="file type that needs to be checked", required=True
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="this flag defines that all files will be checked",
    )
    parser.add_argument(
        "-f",
        "--files",
        nargs="+",
        help="this flag will define which specific files we want to test",
    )
    parser.add_argument(
        "-i", "--index", help="index of which file it is in the directory"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="this flag will print the AST"
    )
    parser.add_argument(
        "-nw",
        "--no-warning",
        action="store_true",
        help="this flag will not print the warnings",
    )
    parser.add_argument(
        "-e",
        "--execute-with",
        help="this flag will execute the mips code with the given program",
        default=None,
        choices=["mars", "spim", "both"],
        type=str.lower,
        required=False,
    )
    parser.add_argument(
        "-s",
        "--silent",
        action="store_true",
        help="this flag will not print the output of the program",
    )
    # disclaimer
    parser.add_argument(
        "-nd",
        "--no-disclaimer",
        action="store_true",
        help="this flag will not print the disclaimer",
    )
    # visualise
    parser.add_argument(
        "-vs",
        "--visualise",
        action="store_true",
        help="this flag will create a dot file and a png file",
    )
    # try:
    args = parser.parse_args()
    filenames = args.files if args.files is not None else []
    if args.files is not None:
        run(
            directory=args.directory,
            file_type=args.type,
            filenames=args.files,
            verbose=args.verbose,
            no_warning=args.no_warning,
            execute_with=ExecuteWith(args.execute_with),
            disclaimer=not args.no_disclaimer,
            silent=args.silent,
            visualise=args.visualise,
        )
    elif args.index is not None:
        files = os.listdir(args.directory)
        run_file(
            directory=args.directory,
            file_type=args.type,
            filename=files[int(args.index)][
                : len(files[int(args.index)]) - len(args.type)
            ],
            verbose=args.verbose,
            no_warning=args.no_warning,
            execute_with=ExecuteWith(args.execute_with),
            disclaimer=not args.no_disclaimer,
            silent=args.silent,
            visualise=args.visualise,
        )
    else:
        filenames = [
            file[: len(file) - len(args.type)]
            for file in os.listdir(args.directory)
            if file.endswith(args.type)
        ]
        run(
            directory=args.directory,
            file_type=args.type,
            filenames=filenames,
            verbose=args.verbose,
            no_warning=args.no_warning,
            execute_with=ExecuteWith(args.execute_with),
            disclaimer=not args.no_disclaimer,
            silent=args.silent,
            visualise=args.visualise,
        )


if __name__ == "__main__":
    main()
