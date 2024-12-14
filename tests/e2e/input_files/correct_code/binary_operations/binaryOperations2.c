#include <stdio.h>

// This should print the number 10 for nested expressions
int main(){
        printf("%d; ", 2*(2+3));
        printf("%f; ", 2*(2.0+3.0));
        printf("%d; ", 2*4+2);
        printf("%f; ", 2.0*4.5+1.0);
        printf("%d; ", 10/2+10/2);
        printf("%f; ", 10.0/2.0+10.0/2.0);
        printf("%d; ", ((100-80)/2)+(5-5));
        printf("%f; ", ((100.0-80.0)/2.0)+(5.0-5.0));

        return 1;
}
