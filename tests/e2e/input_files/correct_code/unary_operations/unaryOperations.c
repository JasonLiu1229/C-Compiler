#include <stdio.h>

// This should print the numbers 9 - 14
int main(){
	int x = 9;
    int a[2]; // [0] = 15, [1] = 12
	printf("%d; ", -(-9)); // 9
    printf("%d; ", ++x); // 10
    a[0] = 15;
	a[1] = 12;
	x = 12;
	printf("%d; ", --a[1]); // 11
    printf("%d; ", x++); // 12
    printf("%d; ", x); // 13
	a[0]--;
    printf("%d; ", a[0]); // 14
    return 1;
}
