#include<stdio.h>

// Prints 10, 9, 9, 7
int main(){
    int a = 10;
    printf("The value of a is %d\n", a);
    a--;
    printf("The value of a is %d\n", a);
    printf("The value of a is %d\n", a--);
    printf("The value of a is %d\n", --a);
}