#include<stdio.h>

int main(){
    int x = 10;
    float y = x + 0.5;
    printf("x is %d\n",x);
    printf("y is %f\n",y);
    while(x<12){
        printf("x is %d\n",x);
        x++;
    }
}