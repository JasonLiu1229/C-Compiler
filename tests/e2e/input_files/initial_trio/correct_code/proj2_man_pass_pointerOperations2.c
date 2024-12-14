int main() {

int x = -60;
int* some_pointer = &x;
*some_pointer = 53;
int** another_pointer = &some_pointer;
int*** triple_pointer = &another_pointer;
int y = ***triple_pointer;
return 0;
}
