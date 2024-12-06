#include<stdio.h>

// Prints 5, 6, 6, 8
int main(){
    int a = 5;
    printf("The value of a is %d\n", a);
    a++;
    printf("The value of a is %d\n", a);
    printf("The value of a is %d\n", a++);
    printf("The value of a is %d\n", ++a);
}