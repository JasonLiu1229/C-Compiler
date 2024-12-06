int main() {

char x = 'a';

char* chr_ptr = &x;

*chr_ptr = 'b';

char another_char = *chr_ptr;
return 0;
}
