int main() {

int x = 478;
int b = -251454;

int* b_ptr = &b;

int** x_ptr = &b_ptr;
x_ptr = &b;
return 0;
}
