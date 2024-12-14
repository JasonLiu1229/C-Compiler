#include <stdio.h>

int main(){
    int day;
    scanf("%d", &day);
    switch (day) {
        case 6:
            printf("Today is Saturday");
            break;
        case 7:
            printf("Today is Sunday");
            break;
        default:
            printf("Looking forward to the Weekend");
    }
}