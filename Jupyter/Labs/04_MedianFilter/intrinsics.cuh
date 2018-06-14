#pragma once

// https://stackoverflow.com/a/36205084/1669370
template <unsigned NumElements, class Compare = void> class StaticSort
{
    template <class A, class C> struct Swap
    {
        template <class T> inline __device__ void s(T &v0, T &v1)
        {
            T t = Compare()(v0, v1) ? v0 : v1;
            v1 = Compare()(v0, v1) ? v1 : v0;
            v0 = t;
        }

        inline __device__ Swap(A &a, const int &i0, const int &i1) { s(a[i0], a[i1]); }
    };

    template <class A> struct Swap <A, void>
    {
        template <class T> inline __device__ void s(T &v0, T &v1)
        {
            T t = v0 < v1 ? v0 : v1; // Min
            v1 = v0 < v1 ? v1 : v0; // Max
            v0 = t;
        }

        __device__ inline Swap(A &a, const int &i0, const int &i1) { s(a[i0], a[i1]); }
    };

    template <class A, class C, int I, int J, int X, int Y> struct PB
    {
        inline __device__ PB(A &a)
        {
            enum { L = X >> 1, M = (X & 1 ? Y : Y + 1) >> 1, IAddL = I + L, XSubL = X - L };
            PB<A, C, I, J, L, M> p0(a);
            PB<A, C, IAddL, J + M, XSubL, Y - M> p1(a);
            PB<A, C, IAddL, J, XSubL, M> p2(a);
        }
    };

    template <class A, class C, int I, int J> struct PB <A, C, I, J, 1, 1>
    {
        inline __device__ PB(A &a) { Swap<A, C> s(a, I - 1, J - 1); }
    };

    template <class A, class C, int I, int J> struct PB <A, C, I, J, 1, 2>
    {
        inline __device__ PB(A &a) { Swap<A, C> s0(a, I - 1, J); Swap<A, C> s1(a, I - 1, J - 1); }
    };

    template <class A, class C, int I, int J> struct PB <A, C, I, J, 2, 1>
    {
        inline __device__  PB(A &a) { Swap<A, C> s0(a, I - 1, J - 1); Swap<A, C> s1(a, I, J - 1); }
    };

    template <class A, class C, int I, int M, bool Stop = false> struct PS
    {
        inline __device__ PS(A &a)
        {
            enum { L = M >> 1, IAddL = I + L, MSubL = M - L };
            PS<A, C, I, L, (L <= 1)> ps0(a);
            PS<A, C, IAddL, MSubL, (MSubL <= 1)> ps1(a);
            PB<A, C, I, IAddL, L, MSubL> pb(a);
        }
    };

    template <class A, class C, int I, int M> struct PS <A, C, I, M, true>
    {
        inline __device__ PS(A &a) {}
    };

public:
    /**
    * Sorts the array/container arr.
    * \param  arr  The array/container to be sorted.
    */
    template <class Container> inline __device__ void operator() (Container &arr) const
    {
        PS<Container, Compare, 1, NumElements, (NumElements <= 1)> ps(arr);
    }

    /**
    * Sorts the array arr.
    * \param  arr  The array to be sorted.
    */
    template <class T> inline __device__ void operator() (T *arr) const
    {
        PS<T*, Compare, 1, NumElements, (NumElements <= 1)> ps(arr);
    }

    template <class T> static inline __device__ void sort(T* arr)
    {
        StaticSort<NumElements, Compare> sorter;
        sorter.operator()(arr);
    }
};



template <typename scalar, int window>
struct medianfilter
{
	static constexpr int size = (window * 2 + 1) * (window * 2 + 1);
	scalar buffer[size];
	scalar work[size];

	__forceinline__ __device__ __host__ void set_Item(int i, scalar val) { buffer[i] = val; }

	__forceinline__  __device__ __host__ scalar apply()
	{
		#pragma unroll
		for (int k = 0; k < size; ++k) {
			work[k] = buffer[k];
		}

		StaticSort<size> sort;
		sort(work);

		return work[size / 2];
	}

	__forceinline__  __device__ __host__ void rollbuffer()
	{
		// TODO
		#pragma unroll
		for (int lj = -window; lj < window; ++lj)
		{
			#pragma unroll
			for (int li = -window; li <= window; ++li)
			{
				buffer[(lj + window)*(window * 2 + 1) + (li + window)] = buffer[(lj + 1 + window)*(window * 2 + 1) + (li + window)];
			}
		}
	}
};
