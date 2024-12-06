int main() {
int x = 5;
float f = 33989.586265;

int z = (int) f;
float z2 = (float) x;

int a = (int) (f + z2 * 2);

float f2 = (float) f;

f2 = (float) (a + z2 / (3*65232));
return 0;
}
