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
        Matrix matrix_naive = {
            .w = TEST_W,
            .h = TEST_H,
            .src = (int *) malloc(sizeof(Matrix) * TEST_W * TEST_H),
            .method = naive_transpose
        };

        for(int i = 0; i < matrix_naive.h; i++)
            for(int j = 0; j< matrix_naive.w; j++)
                *(matrix_naive.src + i * matrix_naive.h + j) = rand();

        clock_gettime(CLOCK_REALTIME, &start);
        matrix_naive.method(matrix_naive.src, matrix_naive.w, matrix_naive.h);
        clock_gettime(CLOCK_REALTIME, &end);
        printf("naive: \t\t\t %ld us\n", diff_in_us(start, end));

    }

    if (matrix_bool[1]) {
        Matrix matrix_sse = {
            .w = TEST_W,
            .h = TEST_H,
            .src = (int *) malloc(sizeof(Matrix) * TEST_W * TEST_H),
            .method = sse_transpose
        };

        for(int i = 0; i < matrix_sse.h; i++)
            for(int j = 0; j< matrix_sse.w; j++)
                *(matrix_sse.src + i * matrix_sse.h + j) = rand();

        clock_gettime(CLOCK_REALTIME, &start);
        matrix_sse.method(matrix_sse.src, matrix_sse.w, matrix_sse.h);
        clock_gettime(CLOCK_REALTIME, &end);
        printf("sse: \t\t\t %ld us\n", diff_in_us(start, end));
    }

    if (matrix_bool[2]) {
        Matrix matrix_sse_pre = {
            .w = TEST_W,
            .h = TEST_H,
            .src = (int *) malloc(sizeof(Matrix) * TEST_W * TEST_H),
            .method = sse_prefetch_transpose
        };

        for(int i = 0; i < matrix_sse_pre.h; i++)
            for(int j = 0; j< matrix_sse_pre.w; j++)
                *(matrix_sse_pre.src + i * matrix_sse_pre.h + j) = rand();

        clock_gettime(CLOCK_REALTIME, &start);
        matrix_sse_pre.method(matrix_sse_pre.src, matrix_sse_pre.w, matrix_sse_pre.h);
        clock_gettime(CLOCK_REALTIME, &end);
        printf("sse_prefetch: \t\t %ld us\n", diff_in_us(start, end));
    }

    return 0;
}
