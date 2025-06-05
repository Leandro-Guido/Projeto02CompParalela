// test_gpu.c
// gcc8 -v -fopenmp -foffload=nvptx-none test_gpu.c -o test_gpu.exe
#include <stdio.h>
#include <omp.h>

int main() {
    int is_device = 0;

    #pragma omp target map(from:is_device)
    {
        is_device = omp_is_initial_device();
    }

    if (is_device)
        printf("Rodando na CPU.\n");
    else
        printf("Rodando na GPU.\n");

    return 0;
}
