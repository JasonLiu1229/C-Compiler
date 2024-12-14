# Register Management
> Register management is the process of assigning variables to registers in a program. This is an important step in the compilation process, as it can have a significant impact on the performance of the compiled program. Register management is a complex problem, and there are many different algorithms that can be used to solve it. This document provides an overview of the different algorithms that can be used for register management, and the data structures that are used to implement them.

## Data Structures
| Data Structure | Advantages                                                                                                    | Disadvantages                                                                          |
|----------------|---------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| Array/List     | O(1) random access to elements by index, memory efficient for small numbers of registers, easily resized      | Inefficient for inserting or deleting elements, fixed size                             |
| Linked List    | Efficient insertion and deletion of elements, can handle an arbitrary number of registers                     | O(n) access time to elements by index, additional memory overhead for storing pointers |
| Stack          | Simple and easy to implement, guaranteed constant time complexity for push and pop operations                 | O(n) access time to elements by index, inefficient for non-LIFO access                 |
| Bitmap         | Efficient use of memory for large numbers of registers, constant time complexity for checking register status | Less efficient for inserting or deleting elements, less intuitive to work with         |

## Algorithms
| Algorithm                          | Advantages                                                                                                                                                         | Disadvantages                                                                                     | Data Structures                                                         |
|------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| Simple Register Allocation         | Simple and easy to implement                                                                                                                                       | Inefficient use of registers, cannot handle programs that use more registers than are available   | Any                                                                     |
| Graph Coloring Register Allocation | Efficient use of registers, can handle programs that use more registers than are available, can be optimized for specific hardware architectures                   | More complex than simple register allocation, can be slow for large programs with many variables  | Arrays, linked lists, bitmaps                                           |
| Linear Scan Register Allocation    | Faster than graph coloring for many programs, can handle programs that use more registers than are available, can be optimized for specific hardware architectures | Less efficient use of registers than graph coloring, more complex than simple register allocation | Any data structure that supports efficient searching for free registers |

### pseudocode for simple register allocation
```
for each variable v in the program:
    if there is a free register:
        assign it to v
    else:
        spill a variable from a register to memory
        assign the now-free register to v
```

### pseudocode for graph coloring register allocation
```
create a graph G where each node is a variable and each edge is a conflict between two variables
color the nodes of G with a set of available registers such that no two adjacent nodes have the same color
if no coloring is possible, spill a variable to memory and retry
```

### pseudocode for linear scan register allocation
```
for each variable v in the program:
    if there is a free register:
        assign it to v and mark it as live
    else:
        find the oldest live variable and spill it to memory
        assign its register to v and mark it as live
    update the liveness information for all variables
```

### Libraries
| Algorithm                          | Description                                                                                                                                                         | Related Libraries |
|------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|
| Simple Register Allocation         | Assigns registers to variables one at a time. If no free registers are available, spills a variable to memory.                                                      | None              |
| Graph Coloring Register Allocation | Creates a graph of variables and conflicts, and colors the nodes with a set of available registers. Can handle programs that use more registers than are available. | LLVM, PyPy        |
| Linear Scan Register Allocation    | Assigns registers based on the liveness of variables at each point in the program. Can handle programs that use more registers than are available.                  | Numba             |

## All libraries
| Library   | Description                                                                                                                                                                                |
|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PyLLVM    | Python binding for the LLVM compiler infrastructure. Provides tools for analyzing and optimizing code, including register allocation.                                                      |
| pycparser | Python library for parsing C code. Can be used to generate an AST of the code, which can then be used for register allocation.                                                             |
| PyPy      | Python interpreter that uses a JIT compiler to optimize code. Includes a built-in register allocator that can be customized and tuned for specific hardware architectures.                 |
| Numba     | Library for JIT compilation of Python code. Includes a register allocator that can be used to optimize numerical code.                                                                     |
| cffi      | Python library for interfacing with C code. Can be used to call C functions that perform register allocation, or to generate Python bindings for C code that includes register allocation. |

### References
| Library | Description | Reference |
| --- | --- | --- |
| PyLLVM | Python binding for the LLVM compiler infrastructure. Provides tools for analyzing and optimizing code, including register allocation. | [GitHub](https://github.com/llvm/llvm-project/tree/main/llvm/bindings/python) |
| pycparser | Python library for parsing C code. Can be used to generate an AST of the code, which can then be used for register allocation. | [GitHub](https://github.com/eliben/pycparser) |
| PyPy | Python interpreter that uses a JIT compiler to optimize code. Includes a built-in register allocator that can be customized and tuned for specific hardware architectures. | [Website](https://www.pypy.org/) |
| Numba | Library for JIT compilation of Python code. Includes a register allocator that can be used to optimize numerical code. | [GitHub](https://github.com/numba/numba) |
| cffi | Python library for interfacing with C code. Can be used to call C functions that perform register allocation, or to generate Python bindings for C code that includes register allocation. | [GitHub](https://github.com/ffi/ffi) |

## Sorting Algorithms
| Algorithm | Description                                                                                                                                                                                                                                                                                               | Reference                                             | Python Library                                                              |
|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------|-----------------------------------------------------------------------------|
| Quicksort | Efficient sorting algorithm that uses a divide-and-conquer strategy to sort elements in-place. Worst-case performance is O(n^2), but average-case performance is O(n log n).                                                                                                                              | [Wikipedia](https://en.wikipedia.org/wiki/Quicksort)  | `sorted()`, `list.sort()`, `numpy.sort()`                                   |
| Mergesort | Stable sorting algorithm that uses a divide-and-conquer strategy to sort elements by merging sublists. Worst-case and average-case performance is O(n log n).                                                                                                                                             | [Wikipedia](https://en.wikipedia.org/wiki/Merge_sort) | `sorted()`, `list.sort()`, `numpy.sort()`                                   |
| Heapsort  | Comparison-based sorting algorithm that builds a heap from the input list and repeatedly extracts the maximum element. Worst-case and average-case performance is O(n log n).                                                                                                                             | [Wikipedia](https://en.wikipedia.org/wiki/Heapsort)   | `heapq.nlargest()`, `heapq.nsmallest()`, `numpy.sort()`                     |
| Timsort   | Hybrid sorting algorithm that uses a combination of mergesort and insertion sort. Designed to perform well on many kinds of real-world data, including partially sorted data and data with small or large elements. Worst-case performance is O(n log n), but average-case performance is closer to O(n). | [Wikipedia](https://en.wikipedia.org/wiki/Timsort)    | `sorted()`, `list.sort()`, `numpy.sort()`, `pandas.DataFrame.sort_values()` |

## Registers for mips
| Register | Alias   | Usage                                                 |
|----------|---------|-------------------------------------------------------|
| $0       | $zero   | Hard-wired to 0                                       |
| $1       | $at     | Reserved for use by the assembler                     |
| $2-$3    | $v0-$v1 | Values for function results and expression evaluation |
| $4-$7    | $a0-$a3 | Arguments for function parameters                     |
| $8-$15   | $t0-$t7 | Temporary data storage                                |
| $16-$23  | $s0-$s7 | Saved temporary data storage                          |
| $24-$25  | $t8-$t9 | Temporary data storage                                |
| $26-$27  | $k0-$k1 | Reserved for use by interrupt/exception handlers      |
| $28      | $gp     | Global pointer                                        |
| $29      | $sp     | Stack pointer                                         |
| $30      | $fp/$s8 | Frame pointer (or saved temporary register)           |
| $31      | $ra     | Return address                                        |


| Operation                        | Example                | Description                                                                                                                                         |
|----------------------------------|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| Arithmetic and Logical Operators |                        |                                                                                                                                                     |
| Add                              | `add $s0, $t0, $t1`    | Adds the values in registers `$t0` and `$t1`, and stores the result in register `$s0`.                                                              |
| Add Immediate                    | `addi $s1, $t2, 100`   | Adds the value 100 to the value in register `$t2`, and stores the result in register `$s1`.                                                         |
| Subtract                         | `sub $s2, $t3, $t4`    | Subtracts the value in register `$t4` from the value in register `$t3`, and stores the result in register `$s2`.                                    |
| Multiply                         | `mult $t5, $t6`        | Multiplies the values in registers `$t5` and `$t6`, and stores the result in two special registers, `$hi` and `$lo`.                                |
| Divide                           | `div $t7, $t8`         | Divides the value in register `$t7` by the value in register `$t8`, and stores the quotient in register `$t9`, and the remainder in register `$t8`. |
| And                              | `and $s3, $t9, $s4`    | Performs a bitwise AND of the values in registers `$t9` and `$s4`, and stores the result in register `$s3`.                                         |
| Or                               | `or $s5, $t0, $s6`     | Performs a bitwise OR of the values in registers `$t0` and `$s6`, and stores the result in register `$s5`.                                          |
| Exclusive Or                     | `xor $s7, $t1, $s8`    | Performs a bitwise exclusive OR of the values in registers `$t1` and `$s8`, and stores the result in register `$s7`.                                |
| Nor                              | `nor $t2, $t3, $t4`    | Performs a bitwise NOR of the values in registers `$t3` and `$t4`, and stores the result in register `$t2`.                                         |
| Set Less Than                    | `slt $t5, $t6, $t7`    | Compares the values in registers `$t6` and `$t7`, and stores the result (either 0 or 1) in register `$t5`.                                          |
| Shift Left Logical               | `sll $s0, $t0, 3`      | Shifts the value in register `$t0` to the left by 3 bits, and stores the result in register `$s0`.                                                  |
| Shift Right Logical              | `srl $s1, $t1, 2`      | Shifts the value in register `$t1` to the right by 2 bits, and stores the result in register `$s1`.                                                 |
| Load and Store Operators         |                        |                                                                                                                                                     |
| Load Word                        | `lw $t0, 0($s0)`       | Loads a 32-bit word from memory at the address `$s0 + 0`, and stores the result in register `$t0`.                                                  |
| Store Word                       | `sw $t1, 4($s1)`       | Stores the 32-bit value in register `$t1` to memory at the address `$s1 + 4`.                                                                       |
| Control Flow Operators           |                        |                                                                                                                                                     |
| Branch If Equal                  | `beq $t2, $t3, Label1` | Compares the values in registers `$t2` and `$t3`. If they are equal, the program branches to the label                                              |