__global__ void helloGPU() {
    printf("Hello World!");
}

int main(void) {
    helloGPU<<<1, 1>>>();
    return 0;
}