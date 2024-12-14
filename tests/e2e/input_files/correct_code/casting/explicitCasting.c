
// Should NOT generate casting warning
int main(){
    float x = 0.5;
    int y = (int)x;
    int z = (int)(x + 0.5);
    printf("%d %d %d\n", (int)x, y, z);
    return 1;
}
