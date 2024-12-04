import argparse
import contextlib
import os

from antlr4 import CommonTokenStream, FileStream

from antlr4_output.FileLexer import FileLexer
from antlr4_output.FileParser import FileParser

from .ast import ErrorListener
from .ast_creator import AstCreator
from .mips import MIPS


def run(
    directory: str,
    file_type: str,
    filenames: list,
    verbose: bool = True,
    no_warning: bool = False,
    execute_with: str | None = None,
    disclaimer: bool = True,
    silent: bool = False,
    visualise: bool = False,
):
    for filename in filenames:
        try:
            # code in blue
            print(f"\033[94m>>> Parsing {directory}{filename}{file_type}\033[0m\n")
            input_stream = FileStream(directory + filename + file_type)
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
            visitor = AstCreator(filename=directory + filename + file_type)
            ast = visitor.visit(parse_tree)
            # handle tree
            ast = visitor.resolve(ast)
            # check if the main function exists
            if not ast.symbolTable.exists("main"):
                raise Exception("No main function found")
            # delete unused variables and generate warning if there are any
            visitor.warnings += ast.delete_unused_variables(visitor.file_name)
            # print the ast if verbose is true
            if verbose and not silent:
                ast.print(4, True, filename)
                ast.symbolTable.print(True)
            # print warnings if there are any warnings and no_warning is false
            if not no_warning and not silent:
                visitor.warn()
            # create dot file
            if visualise:
                ast.dot_language(filename, visitor.symbol_table)
            # generator = LLVM(ast,  "../Output/" + filename + ".ll")
            # generator.convert()
            # generator.execute()

            # create mips code
            generator = MIPS(ast, "../MIPS_output/" + filename + ".asm")
            generator.convert()
            # execute mips code
            if execute_with is not None:
                generator.execute(execute_with, disclaimer, silent)
            # code in green if no errors
            # with open(f"../MIPS_output/logs/{filename}.log.txt", f"{'w' if disclaimer else 'a'}") as f:
            #     f.write(f"Finished execution with exit code 0\n")
            print("\033[92m\n>>> Finished execution with exit code 0\033[0m\n")
        except Exception as e:
            # code in red if error
            # log the error
            with contextlib.suppress(Exception):
                if verbose and not silent:
                    ast.print(4, True, filename)
                    ast.symbolTable.print(True)

            print(f"\033[91m\n>>> Error: {e}\033[0m\n")
            with open(f"../MIPS_output/logs/{filename}.log.txt", "w") as f:
                f.write(f"Error: {e}\n")
            continue


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
            execute_with=args.execute_with,
            disclaimer=not args.no_disclaimer,
            silent=args.silent,
            visualise=args.visualise,
        )
    elif args.index is not None:
        files = os.listdir(args.directory)
        files_one = [
            files[int(args.index) - 1][
                : len(files[int(args.index) - 1]) - len(args.type)
            ]
        ]
        run(
            directory=args.directory,
            file_type=args.type,
            filenames=files_one,
            verbose=args.verbose,
            no_warning=args.no_warning,
            execute_with=args.execute_with,
            disclaimer=not args.no_disclaimer,
            silent=args.silent,
            visualise=args.visualise,
        )
    else:
        for file in os.listdir(args.directory):
            if file.endswith(args.type):
                filenames.append(file[: len(file) - len(args.type)])
        run(
            directory=args.directory,
            file_type=args.type,
            filenames=filenames,
            verbose=args.verbose,
            no_warning=args.no_warning,
            execute_with=args.execute_with,
            disclaimer=not args.no_disclaimer,
            silent=args.silent,
            visualise=args.visualise,
        )
    expected_outputs = []
    actual_outputs = []
    # for filename in filenames:
    #     expected_outputs.append(open(f"../Output/Expected_output/{filename}.log.txt", "r").read())
    #     actual_outputs.append(open(f"../MIPS_output/logs/{filename}.log.txt", "r").read())
    # for i in range(len(actual_outputs)):
    #     TestCase().assertEqual(expected_outputs[i], actual_outputs[i])
    # # except Exception as e:
    #     print(f'Excepted with error \"{e}\"')


if __name__ == "__main__":
    main()
