#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <xmmintrin.h>
#include "pmmintrin.h"

#include "impl.c"

static long diff_in_us(struct timespec start,struct timespec end)
{
    struct timespec time;
    if (end.tv_nsec < start.tv_nsec) {
        time.tv_sec = end.tv_sec - start.tv_sec - 1;
        time.tv_nsec = 1000000000 - start.tv_nsec + end.tv_nsec;
    } else {
        time.tv_sec = end.tv_sec - start.tv_sec;
        time.tv_nsec = end.tv_nsec - start.tv_nsec;
    }

    return (time.tv_sec * 1000000 + time.tv_nsec / 1000);
}

int main(int argc, char *argv[])
{
    struct timespec start, end;
    int width, height;
    width = atoi(argv[1]);
    height = atoi(argv[2]);
    FILE *fp;

    fp = fopen("time.txt", "a+w");

    int *src = (int *) malloc(sizeof(int) * width * height);
    int *src_align __attribute__ ((aligned (16))) = (int *) malloc(sizeof(int) * width * height);
    int *sse_out = (int *) malloc(sizeof(int) * width * height);
    int *sse_lddqu_out = (int *) malloc(sizeof(int) * width * height);
    int *sse_lddqu_align_out = (int *) malloc(sizeof(int) * width * height);

    srand(time(NULL));
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            *(src + i * width + j) = rand();

    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            *(src_align + i * width + j) = rand();

    clock_gettime(CLOCK_REALTIME, &start);
    sse_transpose(src, sse_out, width, height);
    clock_gettime(CLOCK_REALTIME, &end);
    fprintf(fp, "%d,%d,%ld", width, height, diff_in_us(start, end));

    clock_gettime(CLOCK_REALTIME, &start);
    sse_transpose_lddqu(src, sse_lddqu_out, width, height);
    clock_gettime(CLOCK_REALTIME, &end);
    fprintf(fp, ",%ld", diff_in_us(start, end));

    clock_gettime(CLOCK_REALTIME, &start);
    sse_transpose_lddqu(src_align, sse_lddqu_align_out, width, height);
    clock_gettime(CLOCK_REALTIME, &end);
    fprintf(fp, ",%ld\n", diff_in_us(start, end));

    fclose(fp);
    free(src);
    free(src_align);
    free(sse_out);
    free(sse_lddqu_out);
    free(sse_lddqu_align_out);
    return 0;
}
