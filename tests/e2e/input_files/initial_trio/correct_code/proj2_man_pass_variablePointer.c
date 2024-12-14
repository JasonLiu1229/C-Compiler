int main() {
int integer = 5;

int* int_ptr = &integer;

int ** ptr_ptr = &int_ptr;


int **another_pointer = ptr_ptr;


int z = integer + 5;

int_ptr = &z;

int* pointer = &z;

int x = *pointer;
int** x_ptr = &int_ptr;


return 0;
}
