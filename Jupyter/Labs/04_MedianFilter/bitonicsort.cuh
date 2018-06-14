#pragma once
#include <hybridizer.cuda.cuh>

template<typename T, typename size = int>
struct bitonicsort_t
{
	__device__ __host__ static void compare(bool up, T* ar, size from, size to)
	{
		int mid = (to + from + 1) / 2; // left part is larger
		int dist = mid - from;
		for (int k = 0; k < mid; ++k)
		{
			if (k + dist == to) break;
			if ((ar[k] > ar[k + dist]) == up)
			{
				int tmp = ar[k];
				ar[k] = ar[k + dist];
				ar[k + dist] = tmp;
			}
		}
	}

	__device__ __host__ static void merge(bool up, T* ar, size from, size to)
	{
		if ((to - from) <= 1) return;
		compare(up, ar, from, to);
		int mid = (to + from + 1) / 2; // left part is larger
		merge(up, ar, from, mid);
		merge(up, ar, mid, to);
	}

	__device__ __host__ static void sort(bool up, T* ar, size from, size to)
	{
		if ((to - from) <= 1) return;
		int mid = (to + from + 1) / 2; // left part is larger
		sort(true, ar, from, mid);
		sort(false, ar, mid, to);
		merge(up, ar, from, to);
	}
};

template<typename T, typename size> __device__ __host__  void bitonicsort(T* ar, size from, size to)
{
	return bitonicsort_t<T, size>::sort(true, ar, from, to);
}
