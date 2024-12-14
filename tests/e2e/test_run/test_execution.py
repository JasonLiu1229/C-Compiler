import os

import pytest

from src.mips import ExecuteWith
from src.run import run_file


@pytest.fixture
def current_directory() -> str:
    return os.path.dirname(os.path.realpath(__file__))


@pytest.fixture
def correct_code() -> str:
    return os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../input_files/correct_code"
    )


@pytest.fixture
def semantic_errors() -> str:
    return os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../input_files/semantic_errors"
    )


@pytest.fixture
def projects_1_to_3() -> str:
    return os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../input_files/initial_trio"
    )


class TestExeuction:
    def test_binary_operations(
        self,
        correct_code: str,
    ) -> None:
        filenames = [
            file[:-2]
            for file in os.listdir(os.path.join(correct_code, "binary_operations"))
            if file.endswith(".c")
        ]
        for file in filenames:
            output = run_file(
                directory=os.path.join(correct_code, "binary_operations"),
                file_type=".c",
                filename=file,
                verbose=False,
                no_warning=False,
                silent=True,
                execute_with=ExecuteWith.MARS,
                disclaimer=False,
            )
            assert output
            assert output.returncode == 0
            assert output.stdout == "10; 10.0; 10; 10.0; 10; 10.0; 10; 10.0; \n"

    def test_modulo(self, correct_code: str) -> None:
        output = run_file(
            directory=os.path.join(correct_code, "modulo"),
            file_type=".c",
            filename="modulo",
            verbose=False,
            no_warning=False,
            silent=True,
            execute_with=ExecuteWith.MARS,
            disclaimer=False,
        )
        assert output
        assert output.returncode == 0
        assert output.stdout == "1\n"

    def test_comparisons(self, correct_code: str) -> None:
        filenames = [
            file[:-2]
            for file in os.listdir(os.path.join(correct_code, "comparisons"))
            if file.endswith(".c")
        ]
        for file in filenames:
            output = run_file(
                directory=os.path.join(correct_code, "comparisons"),
                file_type=".c",
                filename=file,
                verbose=False,
                no_warning=False,
                silent=True,
                execute_with=ExecuteWith.MARS,
            )
            assert output
            assert output.returncode == 0
            assert output.stdout == "1; 0; 1; 0; 1; 0; \n"

    def test_printf(self, correct_code: str) -> None:
        filenames = [
            file[:-2]
            for file in os.listdir(os.path.join(correct_code, "printf"))
            if file.endswith(".c")
        ]
        for file in filenames:
            output = run_file(
                directory=os.path.join(correct_code, "printf"),
                file_type=".c",
                filename=file,
                verbose=False,
                no_warning=False,
                silent=True,
                execute_with=ExecuteWith.MARS,
            )
            assert output
            assert output.returncode == 0

    def test_variables(self, correct_code: str) -> None:
        filenames = [
            file[:-2]
            for file in os.listdir(os.path.join(correct_code, "variables"))
            if file.endswith(".c")
        ]
        for file in filenames:
            output = run_file(
                directory=os.path.join(correct_code, "variables"),
                file_type=".c",
                filename=file,
                verbose=False,
                no_warning=False,
                silent=True,
                execute_with=ExecuteWith.MARS,
            )
            assert output
            assert output.returncode == 0

    def test_unary_operations(self, correct_code: str) -> None:
        filenames = [
            file[:-2]
            for file in os.listdir(os.path.join(correct_code, "unary_operations"))
            if file.endswith(".c")
        ]
        for file in filenames:
            output = run_file(
                directory=os.path.join(correct_code, "unary_operations"),
                file_type=".c",
                filename=file,
                verbose=False,
                no_warning=False,
                silent=True,
                execute_with=ExecuteWith.MARS,
            )
            assert output
            assert output.returncode == 0

    def test_constant_folding(self, correct_code: str) -> None:
        filenames = [
            file[:-2]
            for file in os.listdir(os.path.join(correct_code, "constant_folding"))
            if file.endswith(".c")
        ]
        for file in filenames:
            output = run_file(
                directory=os.path.join(correct_code, "constant_folding"),
                file_type=".c",
                filename=file,
                verbose=False,
                no_warning=False,
                silent=True,
                execute_with=ExecuteWith.MARS,
            )
            assert output
            assert output.returncode == 0

    def test_constant_propagation(self, correct_code: str) -> None:
        filenames = [
            file[:-2]
            for file in os.listdir(os.path.join(correct_code, "constant_propagation"))
            if file.endswith(".c")
        ]
        for file in filenames:
            output = run_file(
                directory=os.path.join(correct_code, "constant_propagation"),
                file_type=".c",
                filename=file,
                verbose=False,
                no_warning=False,
                silent=True,
                execute_with=ExecuteWith.MARS,
            )
            assert output
            assert output.returncode == 0

    def test_if_else(self, correct_code: str) -> None:
        filenames = [
            file[:-2]
            for file in os.listdir(os.path.join(correct_code, "if_else"))
            if file.endswith(".c")
        ]
        for file in filenames:
            output = run_file(
                directory=os.path.join(correct_code, "if_else"),
                file_type=".c",
                filename=file,
                verbose=False,
                no_warning=False,
                silent=True,
                execute_with=ExecuteWith.MARS,
            )
            assert output
            assert output.returncode == 0

    def test_switch(self, correct_code: str) -> None:
        filenames = [
            file[:-2]
            for file in os.listdir(os.path.join(correct_code, "switch"))
            if file.endswith(".c")
        ]
        for file in filenames:
            output = run_file(
                directory=os.path.join(correct_code, "switch"),
                file_type=".c",
                filename=file,
                verbose=False,
                no_warning=False,
                silent=True,
                execute_with=ExecuteWith.MARS,
            )
            assert output
            assert output.returncode == 0

    def test_loops(self, correct_code: str) -> None:
        filenames = [
            file[:-2]
            for file in os.listdir(os.path.join(correct_code, "loops"))
            if file.endswith(".c")
        ]
        for file in filenames:
            output = run_file(
                directory=os.path.join(correct_code, "loops"),
                file_type=".c",
                filename=file,
                verbose=False,
                no_warning=False,
                silent=True,
                execute_with=ExecuteWith.MARS,
            )
            assert output
            assert output.returncode == 0

    def test_break_continue(self, correct_code: str) -> None:
        filenames = [
            file[:-2]
            for file in os.listdir(os.path.join(correct_code, "break_continue"))
            if file.endswith(".c")
        ]
        for file in filenames:
            output = run_file(
                directory=os.path.join(correct_code, "break_continue"),
                file_type=".c",
                filename=file,
                verbose=False,
                no_warning=False,
                silent=True,
                execute_with=ExecuteWith.MARS,
            )
            assert output
            assert output.returncode == 0

    def test_increment_decrement(self, correct_code: str) -> None:
        filenames = [
            file[:-2]
            for file in os.listdir(os.path.join(correct_code, "increment_decrement"))
            if file.endswith(".c")
        ]
        for file in filenames:
            output = run_file(
                directory=os.path.join(correct_code, "increment_decrement"),
                file_type=".c",
                filename=file,
                verbose=False,
                no_warning=False,
                silent=True,
                execute_with=ExecuteWith.MARS,
            )
            assert output
            assert output.returncode == 0

    def test_conversion(self, correct_code: str) -> None:
        filenames = [
            file[:-2]
            for file in os.listdir(os.path.join(correct_code, "conversion"))
            if file.endswith(".c")
        ]
        for file in filenames:
            output = run_file(
                directory=os.path.join(correct_code, "conversion"),
                file_type=".c",
                filename=file,
                verbose=False,
                no_warning=False,
                silent=True,
                execute_with=ExecuteWith.MARS,
            )
            assert output
            assert output.returncode == 0

    def test_forward_declaration(self, correct_code: str) -> None:
        filenames = [
            file[:-2]
            for file in os.listdir(os.path.join(correct_code, "forward_declaration"))
            if file.endswith(".c")
        ]
        for file in filenames:
            output = run_file(
                directory=os.path.join(correct_code, "forward_declaration"),
                file_type=".c",
                filename=file,
                verbose=False,
                no_warning=False,
                silent=True,
                execute_with=ExecuteWith.MARS,
            )
            assert output
            assert output.returncode == 0

    def test_casting(self, correct_code: str) -> None:
        filenames = [
            file[:-2]
            for file in os.listdir(os.path.join(correct_code, "casting"))
            if file.endswith(".c")
        ]
        for file in filenames:
            output = run_file(
                directory=os.path.join(correct_code, "casting"),
                file_type=".c",
                filename=file,
                verbose=False,
                no_warning=False,
                silent=True,
                execute_with=ExecuteWith.MARS,
            )
            assert output
            assert output.returncode == 0

    def test_arguments(self, correct_code: str) -> None:
        filenames = [
            file[:-2]
            for file in os.listdir(os.path.join(correct_code, "arguments"))
            if file.endswith(".c")
        ]
        for file in filenames:
            output = run_file(
                directory=os.path.join(correct_code, "arguments"),
                file_type=".c",
                filename=file,
                verbose=False,
                no_warning=False,
                silent=True,
                execute_with=ExecuteWith.MARS,
            )
            assert output
            assert output.returncode == 0

    def test_scoping(self, correct_code: str) -> None:
        filenames = [
            file[:-2]
            for file in os.listdir(os.path.join(correct_code, "scoping"))
            if file.endswith(".c")
        ]
        for file in filenames:
            output = run_file(
                directory=os.path.join(correct_code, "scoping"),
                file_type=".c",
                filename=file,
                verbose=False,
                no_warning=False,
                silent=True,
                execute_with=ExecuteWith.MARS,
            )
            assert output
            assert output.returncode == 0

    def test_comments(self, correct_code: str) -> None:
        filenames = [
            file[:-2]
            for file in os.listdir(os.path.join(correct_code, "comments"))
            if file.endswith(".c")
        ]
        for file in filenames:
            output = run_file(
                directory=os.path.join(correct_code, "comments"),
                file_type=".c",
                filename=file,
                verbose=False,
                no_warning=False,
                silent=True,
                execute_with=ExecuteWith.MARS,
            )
            assert output
            assert output.returncode == 0

    def test_dereferencing(self, correct_code: str) -> None:
        filenames = [
            file[:-2]
            for file in os.listdir(os.path.join(correct_code, "dereferencing"))
            if file.endswith(".c")
        ]
        for file in filenames:
            output = run_file(
                directory=os.path.join(correct_code, "dereferencing"),
                file_type=".c",
                filename=file,
                verbose=False,
                no_warning=False,
                silent=True,
                execute_with=ExecuteWith.MARS,
            )
            assert output
            assert output.returncode == 0

    def test_recursion(self, correct_code: str) -> None:
        filenames = [
            file[:-2]
            for file in os.listdir(os.path.join(correct_code, "recursion"))
            if file.endswith(".c")
        ]
        for file in filenames:
            output = run_file(
                directory=os.path.join(correct_code, "recursion"),
                file_type=".c",
                filename=file,
                verbose=False,
                no_warning=False,
                silent=True,
                execute_with=ExecuteWith.MARS,
            )
            assert output
            assert output.returncode == 0

    @pytest.mark.skip(reason="scanf input has not been implemented")
    def test_scanf(self, correct_code: str) -> None:
        filenames = [
            file[:-2]
            for file in os.listdir(os.path.join(correct_code, "scanf"))
            if file.endswith(".c")
        ]
        for file in filenames:
            output = run_file(
                directory=os.path.join(correct_code, "scanf"),
                file_type=".c",
                filename=file,
                verbose=False,
                no_warning=False,
                silent=True,
                execute_with=ExecuteWith.MARS,
            )
            assert output
            assert output.returncode == 0
