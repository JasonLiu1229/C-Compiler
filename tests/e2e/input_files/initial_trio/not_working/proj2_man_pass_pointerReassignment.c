int main() {
const int b = -6250;
const int x = 5;

int* non_const_pointer = &x;

*non_const_pointer = 36941;

non_const_pointer = &b;


char c = 'x';
char nl = '\n';

char* char_ptr = &c;

*char_ptr = '\t';
char_ptr = &nl;
return 0;
}
