#include <stdio.h>

// expected output: a = 0 a= 1
int main(){
    int a = 0;
    while (1){
        switch (a) {
            case 0:
                printf("a = 0\n");
                a++;
                break;
            case 1:
                printf("a = 1\n");
                a++;
                break;
            case 2:
                a++;
                break;
            case 3:
                printf("Something went wrong\n");
                a++;
                break;
            default:
                printf("a = %d\n", a);
                a++;
                break;
        }
        if (a == 3){
            break;
        }
    }
    return 0;
}