int main() {

int x = 4;
int b = 9632;

const int* x_ptr = &b;
*x_ptr = x;

return 0;
}
