int main() {

int x = 4;
int b = 9632;

const int* x_ptr = &x;
x_ptr = &b;

return 0;
}
