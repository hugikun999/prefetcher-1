#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <stdbool.h>

#include <xmmintrin.h>
#include "pmmintrin.h"
#include "impl_interface.c"

#define TEST_H 4096
#define TEST_W 4096

#define TEST_MATRIX(name) \
	Matrix matrix_##name = { \
		.w = TEST_W, \
		.h = TEST_H, \
		.src = (int *) malloc(sizeof(Matrix) * TEST_W * TEST_H), \
		.method = name##_transpose \
	}; \
	for(int i = 0; i < matrix_##name.h; i++) \
		for(int j = 0; j < matrix_##name.w; j++) \
			*(matrix_##name.src + i * matrix_##name.h + j) = rand(); \
	clock_gettime(CLOCK_REALTIME, &start); \
	matrix_##name.method(matrix_##name.src, matrix_##name.w, matrix_##name.h); \
	clock_gettime(CLOCK_REALTIME, &end);


//0: naive	1:sse	2:sse_prefetch
static bool matrix_bool[3] = {false, false, false};

static long diff_in_us(struct timespec t1, struct timespec t2)
{
    struct timespec diff;
    if (t2.tv_nsec - t1.tv_nsec < 0) {
        diff.tv_sec = t2.tv_sec - t1.tv_sec - 1;
        diff.tv_nsec = t2.tv_nsec - t1.tv_nsec + 1000000000;
    } else {
        diff.tv_sec = t2.tv_sec - t1.tv_sec;
        diff.tv_nsec = t2.tv_nsec - t1.tv_nsec;
    }

    return (diff.tv_sec * 1000000.0 + diff.tv_nsec / 1000.0);
}

int main(int argc, char *argv[])
{
    option_init(&argc, &argv, matrix_bool);
    struct timespec start, end;
    srand(time(NULL));

    if (matrix_bool[0]) {
        TEST_MATRIX(naive);
        printf("naive: \t\t\t %ld us\n", diff_in_us(start, end));
    }

    if (matrix_bool[1]) {
        TEST_MATRIX(sse);
        printf("sse: \t\t\t %ld us\n", diff_in_us(start, end));
    }

    if (matrix_bool[2]) {
        TEST_MATRIX(prefetch);
        printf("sse_prefetch: \t\t %ld us\n", diff_in_us(start, end));
    }

    return 0;
}
