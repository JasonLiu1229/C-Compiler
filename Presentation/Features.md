# All working features

Common parameter settings:

| Short command | Long command    | Type     | Required                  | Description                                                                      |
| ------------- | --------------- | -------- | ------------------------- | -------------------------------------------------------------------------------- |
| -d            | --directory     | `<path>` | **Yes**                   | Directory of the input files                                                     |
| -t            | --type          | `<str>`  | **Yes**                   | File extension                                                                   |
| -f            | --files         | `[str]`  | Exclusive. Takes priority | Files to parse                                                                   |
| -a            | --all           | `NaN`    | Exclusive. Least priority | Parse all the files in the directory                                             |
| -i            | --index         | `<int>`  | Exclusive                 | Parse the i-th file in the directory                                             |
| -v            | --verbose       | `NaN`    | No                        | Print the AST                                                                    |
| -nw           | --no-warning    | `NaN`    | No                        | Do not print warnings                                                            |
| -e            | --execute-with  | `<str>`  | No                        | Execute the mips code generated for the program. Can be "spim", "mars" or "both" |
| -s            | --silent        | `NaN`    | No                        | Do not print the output of the program                                           |
| -nd           | --no-disclaimer | `NaN`    | No                        | Remove the disclaimers for "spim" and "mars" from the output                     |
| -vs           | --visualise     | `NaN`    | No                        | Generate a dot format file and compile it to a png                               |
| -h            | --help          | `NaN`    | `NaN`                     | For help                                                                         |

Example code with specified input files:

python3 -d ../input_files/ -t .c -f Project3 Project2 ...

Example code with only the input directory and file extension

python3 -d ../input_files/ -t .c -a

Example code with only the input directory and an index

python3 -d ../input_files/ -t .c -i 3

## Features
>
>These are all the working features of the project.

Compile all `.c` files.

```shell
python3 ../src/run.py -d ../input_files/CorrectCode/fully_working/ -t .c -a
```

The output of the scripts is located in the `MIPS_output` folder.

### Mandatory features

#### Correct code
>
> Binary operations

```shell
python3 ../src/run.py -d ../input_files/CorrectCode/fully_working/ -t .c -f binaryOperations1 \
binaryOperations2 -v -e spim
```

[binaryOperations1.c](../input_files/CorrectCode/fully_working/binaryOperations1.c) and
[binaryOperations2.c](../input_files/CorrectCode/fully_working/binaryOperations2.c)  
[binaryOperations1.asm](../MIPS_output/binaryOperations1.asm) and
[binaryOperations2.asm](../MIPS_output/binaryOperations2.asm)

> Comparison operations

```shell
python3 ../src/run.py -d ../input_files/CorrectCode/fully_working/ -t .c -f comparisons1 \
comparisons2 -e spim -nd
```

[comparisons1.c](../input_files/CorrectCode/fully_working/comparisons1.c) and
[comparisons2.c](../input_files/CorrectCode/fully_working/comparisons2.c)  
[comparisons1.asm](../MIPS_output/comparisons1.asm) and
[comparisons2.asm](../MIPS_output/comparisons2.asm)
> Unary operations

```shell
python3 ../src/run.py -d ../input_files/CorrectCode/fully_working/ -t .c -f unaryOperations \
-e spim -nd
```

[unaryOperations.c](../input_files/CorrectCode/fully_working/unaryOperations.c)  
[unaryOperations.asm](../MIPS_output/unaryOperations.asm)
> Constant folding

```shell
python3 ../src/run.py -d ../input_files/CorrectCode/fully_working/ -t .c -f constantFolding constantFolding2 -e spim -nd
```

[constantFolding.c](../input_files/CorrectCode/fully_working/constantFolding.c) and
[constantFolding2.c](../input_files/CorrectCode/fully_working/constantFolding2.c)  
[constantFolding.asm](../MIPS_output/constantFolding.asm) and
[constantFolding2.asm](../MIPS_output/constantFolding2.asm)

> Variables
>> Types: int, float, char

```shell
python3 ../src/run.py -d ../input_files/CorrectCode/fully_working/ -t .c -f variables1 \
variables2 variables5 -e spim -nd
```

[variables1.c](../input_files/CorrectCode/fully_working/variables1.c),
[variables2.c](../input_files/CorrectCode/fully_working/variables2.c) and
[variables5.c](../input_files/CorrectCode/fully_working/variables5.c)  
[variables1.asm](../MIPS_output/variables1.asm),
[variables2.asm](../MIPS_output/variables2.asm) and
[variables5.asm](../MIPS_output/variables5.asm)
>> Pointers

```shell
python3 ../src/run.py -d ../input_files/CorrectCode/fully_working/ -t .c -f variables4 \
variables6 dereferenceAssignment -e spim -nd
```

[variables4.c](../input_files/CorrectCode/fully_working/variables4.c),
[variables6.c](../input_files/CorrectCode/fully_working/variables6.c) and
[dereferenceAssignment.c](../input_files/CorrectCode/fully_working/dereferenceAssignment.c)  
[variables4.asm](../MIPS_output/variables4.asm),
[variables6.asm](../MIPS_output/variables6.asm) and
[dereferenceAssignment.asm](../MIPS_output/dereferenceAssignment.asm)
>> Arrays

```shell
python3 ../src/run.py -d ../input_files/CorrectCode/fully_working/ -t .c -f variables3 \
variables7 variables8 -e spim -nd
```

[variables3.c](../input_files/CorrectCode/fully_working/variables3.c),
[variables7.c](../input_files/CorrectCode/fully_working/variables7.c) and
[variables8.c](../input_files/CorrectCode/fully_working/variables8.c)  
[variables3.asm](../MIPS_output/variables3.asm),
[variables7.asm](../MIPS_output/variables7.asm) and
[variables8.asm](../MIPS_output/variables8.asm)
>> Const variables

```shell
python3 ../src/run.py -d ../input_files/CorrectCode/fully_working/ -t .c -f variableConst -e spim -nd
```

[variableConst.c](../input_files/CorrectCode/fully_working/variableConst.c)  
[variableConst.asm](../MIPS_output/variableConst.asm)
> Conditionals: if, else, scopes

```shell
python3 ../src/run.py -d ../input_files/CorrectCode/fully_working/ -t .c -f if ifElse \
scoping  -e spim -nd
```

[if.c](../input_files/CorrectCode/fully_working/if.c),
[ifElse.c](../input_files/CorrectCode/fully_working/ifElse.c) and
[scoping.c](../input_files/CorrectCode/fully_working/scoping.c)  
[if.asm](../MIPS_output/if.asm),
[ifElse.asm](../MIPS_output/ifElse.asm) and
[scoping.asm](../MIPS_output/scoping.asm)
> Loops: while, for

```shell
python3 ../src/run.py -d ../input_files/CorrectCode/fully_working/ -t .c -f for while -e spim -nd
```

[for.c](../input_files/CorrectCode/fully_working/for.c) and
[while.c](../input_files/CorrectCode/fully_working/while.c)  
[for.asm](../MIPS_output/for.asm) and
[while.asm](../MIPS_output/while.asm)
> Functions
>> Forward declaration

```shell
python3 ../src/run.py -d ../input_files/CorrectCode/fully_working/ -t .c -f \
forwardDeclaration -e spim -nd
```

[forwardDeclaration.c](../input_files/CorrectCode/fully_working/forwardDeclaration.c)  
[forwardDeclaration.asm](../MIPS_output/forwardDeclaration.asm)
>> Recursive

```shell
python3 ../src/run.py -d ../input_files/CorrectCode/fully_working/ -t .c -f fibonacciRecursive \
-e spim
```

[fibonacciRecursive.c](../input_files/CorrectCode/fully_working/fibonacciRecursive.c)  
[fibonacciRecursive.asm](../MIPS_output/fibonacciRecursive.asm)
>> General tests

```shell
python3 ../src/run.py -d ../input_files/CorrectCode/fully_working/ -t .c -f pointerArgument \
prime -e spim
```

[pointerArgument.c](../input_files/CorrectCode/fully_working/pointerArgument.c) and
[prime.c](../input_files/CorrectCode/fully_working/prime.c)  
[pointerArgument.asm](../MIPS_output/pointerArgument.asm) and
[prime.asm](../MIPS_output/prime.asm)
> Printf

```shell
python3 ../src/run.py -d ../input_files/CorrectCode/fully_working/ -t .c -f printf1 printf2 \
printf3 -e spim -nd
```

[printf1.c](../input_files/CorrectCode/fully_working/printf1.c),
[printf2.c](../input_files/CorrectCode/fully_working/printf2.c) and
[printf3.c](../input_files/CorrectCode/fully_working/printf3.c)  
[printf1.asm](../MIPS_output/printf1.asm),
[printf2.asm](../MIPS_output/printf2.asm) and
[printf3.asm](../MIPS_output/printf3.asm)
> Scanf

```shell
python3 ../src/run.py -d ../input_files/CorrectCode/fully_working/ -t .c -f scanf1 scanf2 \
-e spim
```

[scanf1.c](../input_files/CorrectCode/fully_working/scanf1.c) and
[scanf2.c](../input_files/CorrectCode/fully_working/scanf2.c)  
[scanf1.asm](../MIPS_output/scanf1.asm) and
[scanf2.asm](../MIPS_output/scanf2.asm)

#### Semantic errors
>
> Incompatible types
>> Working because of conversions

```shell
python3 ../src/run.py -d ../input_files/SemanticErrors/fully_working/ -t .c -f \
incompatibleTypes1 incompatibleTypes2 incompatibleTypes4 incompatibleTypes5 \
incompatibleTypes6 incompatibleTypes7  -e spim -nd
```

[incompatibleTypes1.c](../input_files/SemanticErrors/fully_working/incompatibleTypes1.c),
[incompatibleTypes2.c](../input_files/SemanticErrors/fully_working/incompatibleTypes2.c),
[incompatibleTypes4.c](../input_files/SemanticErrors/fully_working/incompatibleTypes4.c),
[incompatibleTypes5.c](../input_files/SemanticErrors/fully_working/incompatibleTypes5.c),
[incompatibleTypes6.c](../input_files/SemanticErrors/fully_working/incompatibleTypes6.c) and
[incompatibleTypes7.c](../input_files/SemanticErrors/fully_working/incompatibleTypes7.c)  
[incompatibleTypes1.asm](../MIPS_output/incompatibleTypes1.asm),
[incompatibleTypes2.asm](../MIPS_output/incompatibleTypes2.asm),
[incompatibleTypes4.asm](../MIPS_output/incompatibleTypes4.asm),
[incompatibleTypes5.asm](../MIPS_output/incompatibleTypes5.asm),
[incompatibleTypes6.asm](../MIPS_output/incompatibleTypes6.asm) and
[incompatibleTypes7.asm](../MIPS_output/incompatibleTypes7.asm)
>> Errors, as intended

```shell
python3 ../src/run.py -d ../input_files/SemanticErrors/fully_working/ -t .c -f \
incompatibleTypes3  -e spim -nd
```

[incompatibleTypes3.c](../input_files/SemanticErrors/fully_working/incompatibleTypes3.c)
> Undeclared variables

```shell
python3 ../src/run.py -d ../input_files/SemanticErrors/fully_working/ -t .c -f \
undeclaredVariable1 undeclaredVariable2 undeclaredVariable3  -e spim -nd
```

[undeclaredVariable1.c](../input_files/SemanticErrors/fully_working/undeclaredVariable1.c),
[undeclaredVariable2.c](../input_files/SemanticErrors/fully_working/undeclaredVariable2.c) and
[undeclaredVariable3.c](../input_files/SemanticErrors/fully_working/undeclaredVariable3.c)
> Redeclared variables

```shell
python3 ../src/run.py -d ../input_files/SemanticErrors/fully_working/ -t .c -f \
variableRedefinition1 variableRedefinition2 variableRedefinition3 variableRedefinition4 \
variableRedefinition5 variableRedefinition6 -e spim -nd
```

[variableRedefinition1.c](../input_files/SemanticErrors/fully_working/variableRedefinition1.c),
[variableRedefinition2.c](../input_files/SemanticErrors/fully_working/variableRedefinition2.c),
[variableRedefinition3.c](../input_files/SemanticErrors/fully_working/variableRedefinition3.c),
[variableRedefinition4.c](../input_files/SemanticErrors/fully_working/variableRedefinition4.c),
[variableRedefinition5.c](../input_files/SemanticErrors/fully_working/variableRedefinition5.c) and
[variableRedefinition6.c](../input_files/SemanticErrors/fully_working/variableRedefinition6.c)
> Rvalue assignment

```shell
python3 ../src/run.py -d ../input_files/SemanticErrors/fully_working/ -t .c -f \
rvalueAssignment1 rvalueAssignment2 -e spim -nd
```

> Const reassignment

```shell
python3 ../src/run.py -d ../input_files/SemanticErrors/fully_working/ -t .c -f \
constReassignment -e spim -nd
```

> Undeclared functions

```shell
python3 ../src/run.py -d ../input_files/SemanticErrors/fully_working/ -t .c -f \
undeclaredFunction -e spim -nd
```

[undeclaredFunction.c](../input_files/SemanticErrors/fully_working/undeclaredFunction.c)
> Redeclared functions

```shell
python3 ../src/run.py -d ../input_files/SemanticErrors/fully_working/ -t .c -f \
declarationDeclarationMismatch1 declarationDeclarationMismatch2 \
declarationDeclarationMismatch3 -e spim -nd
```

[declarationDeclarationMismatch1.c](../input_files/SemanticErrors/fully_working/declarationDeclarationMismatch1.c),
[declarationDeclarationMismatch2.c](../input_files/SemanticErrors/fully_working/declarationDeclarationMismatch2.c) and
[declarationDeclarationMismatch3.c](../input_files/SemanticErrors/fully_working/declarationDeclarationMismatch3.c)
> Redefined functions

```shell
python3 ../src/run.py -d ../input_files/SemanticErrors/fully_working/ -t .c -f \
declarationDefinitionMismatch1 declarationDefinitionMismatch2 \
declarationDefinitionMismatch3 functionRedefinition1 functionRedefinition2 \
 functionRedefinition3 -e spim -nd
```

[declarationDefinitionMismatch1.c](../input_files/SemanticErrors/fully_working/declarationDefinitionMismatch1.c),
[declarationDefinitionMismatch2.c](../input_files/SemanticErrors/fully_working/declarationDefinitionMismatch2.c),
[declarationDefinitionMismatch3.c](../input_files/SemanticErrors/fully_working/declarationDefinitionMismatch3.c),
[functionRedefinition1.c](../input_files/SemanticErrors/fully_working/functionRedefinition1.c),
[functionRedefinition2.c](../input_files/SemanticErrors/fully_working/functionRedefinition2.c) and
[functionRedefinition3.c](../input_files/SemanticErrors/fully_working/functionRedefinition3.c)
> Definition in local scope

```shell
python3 ../src/run.py -d ../input_files/SemanticErrors/fully_working/ -t .c -f \
definitionInLocalScope -e spim -nd
```

[definitionInLocalScope.c](../input_files/SemanticErrors/fully_working/definitionInLocalScope.c)
> Parameter mismatch

```shell
python3 ../src/run.py -d ../input_files/SemanticErrors/fully_working/ -t .c -f \
functionCallargumentmismatch1 functionCallargumentmismatch2 functionCallargumentmismatch3 \
functionCallargumentmismatch4 -e spim -nd
```

[functionCallargumentMismatch1.c](../input_files/SemanticErrors/fully_working/functionCallargumentMismatch1.c),
[functionCallargumentMismatch2.c](../input_files/SemanticErrors/fully_working/functionCallargumentMismatch2.c),
[functionCallargumentMismatch3.c](../input_files/SemanticErrors/fully_working/functionCallargumentMismatch3.c) and
[functionCallargumentMismatch4.c](../input_files/SemanticErrors/fully_working/functionCallargumentMismatch4.c)
> Parameter redefinition

```shell
python3 ../src/run.py -d ../input_files/SemanticErrors/fully_working/ -t .c -f \
parameterRedefinition1 parameterRedefinition2 parameterRedefinition3 -e spim -nd
```

[parameterRedefinition1.c](../input_files/SemanticErrors/fully_working/parameterRedefinition1.c),
[parameterRedefinition2.c](../input_files/SemanticErrors/fully_working/parameterRedefinition2.c) and
[parameterRedefinition3.c](../input_files/SemanticErrors/fully_working/parameterRedefinition3.c)
> Return type mismatch

```shell
python3 ../src/run.py -d ../input_files/SemanticErrors/fully_working/ -t .c -f \
returnTypeMismatch -e spim -nd
```

[returnTypeMismatch.c](../input_files/SemanticErrors/fully_working/returnTypeMismatch.c)
> Invalid include

```shell
python3 ../src/run.py -d ../input_files/SemanticErrors/fully_working/ -t .c -f \
invalidIncludeError -e spim -nd
```

[invalidIncludeError.c](../input_files/SemanticErrors/fully_working/invalidIncludeError.c)
> Main not found

```shell
python3 ../src/run.py -d ../input_files/SemanticErrors/fully_working/ -t .c -f mainNotFound -e spim -nd
```

[mainNotFound.c](../input_files/SemanticErrors/fully_working/mainNotFound.c)
> Return outside function

```shell
python3 ../src/run.py -d ../input_files/SemanticErrors/fully_working/ -t .c -f returnOutsideFunction -e spim -nd
```

[returnOutsideFunction.c](../input_files/SemanticErrors/fully_working/returnOutsideFunction.c)
> Dereference type mismatch

```shell
python3 ../src/run.py -d ../input_files/SemanticErrors/fully_working/ -t .c -f dereferenceTypeMismatch1 \
dereferenceTypeMismatch2 -e spim -nd
```

[dereferenceTypeMismatch1.c](../input_files/SemanticErrors/fully_working/dereferenceTypeMismatch1.c) and
[dereferenceTypeMismatch2.c](../input_files/SemanticErrors/fully_working/dereferenceTypeMismatch2.c)
> Array errors

```shell
python3 ../src/run.py -d ../input_files/SemanticErrors/fully_working/ -t .c -f arrayAccesstypeMismatch \
arrayAccesstypeMismatch2 arrayCompareError arraySizeTypeMismatch -e spim -nd
```

[arrayAccessTypeMismatch.c](../input_files/SemanticErrors/fully_working/arrayAccessTypeMismatch.c),
[arrayAccessTypeMismatch2.c](../input_files/SemanticErrors/fully_working/arrayAccessTypeMismatch2.c),
[arrayCompareError.c](../input_files/SemanticErrors/fully_working/arrayCompareError.c) and
> Continue and break outside loop

```shell
python3 ../src/run.py -d ../input_files/SemanticErrors/fully_working/ -t .c -f \
invalidLoopControlStatement -e spim -nd
```

[invalidLoopControlStatement.c](../input_files/SemanticErrors/fully_working/invalidLoopControlStatement.c)
> Invalid operations

```shell
python3 ../src/run.py -d ../input_files/SemanticErrors/fully_working/ -t .c -f invalidUnaryOperation \
pointerOperationError -e spim -nd
```

[invalidUnaryOperation.c](../input_files/SemanticErrors/fully_working/invalidUnaryOperation.c) and
[pointerOperationError.c](../input_files/SemanticErrors/fully_working/pointerOperationError.c)

### Optional features
>
> Implicit casting

```shell
python3 ../src/run.py -d ../input_files/CorrectCode/fully_working/ -t .c -f floatToIntConversion \
intToFloatconversion  -e spim -nd
```

[floatToIntConversion.c](../input_files/CorrectCode/fully_working/floatToIntConversion.c) and
[intToFloatConversion.c](../input_files/CorrectCode/fully_working/intToFloatConversion.c)  
[floatToIntConversion.asm](../MIPS_output/floatToIntConversion.asm)
[intToFloatConversion.asm](../MIPS_output/intToFloatConversion.asm)
> Modulo

```shell
python3 ../src/run.py -d ../input_files/CorrectCode/fully_working/ -t .c -f modulo -e spim -nd
```

[modulo.c](../input_files/CorrectCode/fully_working/modulo.c)  
[modulo.asm](../MIPS_output/modulo.asm)
> Increment and decrement

```shell
python3 ../src/run.py -d ../input_files/CorrectCode/fully_working/ -t .c -f increment \
decrement -e spim -nd
```

[increment.c](../input_files/CorrectCode/fully_working/increment.c) and
[decrement.c](../input_files/CorrectCode/fully_working/decrement.c)  
[increment.asm](../MIPS_output/increment.asm) and
[decrement.asm](../MIPS_output/decrement.asm)
> Switch

```shell
python3 ../src/run.py -d ../input_files/CorrectCode/fully_working/ -t .c -f switch1 switch2 -e spim -nd
```

[switch1.c](../input_files/CorrectCode/fully_working/switch1.c) and
[switch2.c](../input_files/CorrectCode/fully_working/switch2.c)  
[switch1.asm](../MIPS_output/switch1.asm) and
[switch2.asm](../MIPS_output/switch2.asm)
> No code for unused variables

```shell
python3 ../src/run.py -d ../input_files/CorrectCode/fully_working/ -t .c -f unusedVariables -e spim -nd
```

[unusedVariables.c](../input_files/CorrectCode/fully_working/unusedVariables.c)  
[unusedVariables.asm](../MIPS_output/unusedVariables.asm)
> Comments

```shell
python3 ../src/run.py -d ../input_files/CorrectCode/fully_working/ -t .c -f comments -e spim -nd
```

[comments.c](../input_files/CorrectCode/fully_working/comments.c)
[comments.asm](../MIPS_output/comments.asm)

## Authors

| **Name**   | **Email**                             | **Student number** |
| ---------- | ------------------------------------- | ------------------ |
| _Jason L._ | <Jason.Liu@student.uantwerpen.be>     | 20213082           |
| _Orfeo T._ | Orfeo.Terkuç<i@student.uantwerpen.be> | 20213863           |
