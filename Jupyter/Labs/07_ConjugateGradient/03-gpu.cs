using System;
using System.Threading.Tasks;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using Hybridizer.Runtime.CUDAImports;

namespace Hybridizer.Basic.Maths
{
    unsafe class Program
    {
        static HybRunner runner;
        static dynamic wrapper;
        
        static void Main(string[] args)
        {
            int size = 1000000; // very slow convergence with no preconditioner
            SparseMatrix A = SparseMatrix.Laplacian_1D(size);
            //initialise wrapper and Hybrunner
            float[] B = new float[size];
            float[] X = new float[size];

            int maxiter = 1000;
            float eps = 1.0e-09f;

            for (int i = 0; i < size; ++i)
            {
                B[i] = 1.0f; // right side
                X[i] = 0.0f; // starting point
            }

            ConjugateGradient(X, A, B, maxiter, eps);

            Console.Out.WriteLine("OK!");
        }

        public static void ConjugateGradient(float[] X, SparseMatrix A, float[] B, int maxiter, float eps)
        {
            int N = (int)B.Length;
            float[] R = new float[N];
            float[] P = new float[N];
            float[] AP = new float[N];

            Fmsub(R, B, A, X, N);                       // R = B - A*X
            Copy(P, R, N);
            int k = 0;
            while (k < maxiter)
            {
                /*
                 * Call methods with the wrapper
                 */
                Multiply(AP, A, P, N);                  // AP = A*P
                float[] r = new float[] { 0.0f };
                ScalarProd(N, R, R, r);                  // save <R|R>
                float[] rScalar = new float[] { 0.0f };
                ScalarProd(N, P, AP, rScalar);
                float alpha = r[0] / rScalar[0];         // alpha = <R|R> / <P|AP>
                Saxpy(X, X, alpha, P, N);               // X = X - alpha*P
                Saxpy(R, R, -alpha, AP, N);             // RR = R-alpha*AP
                float[] rr = new float[] { 0.0f };
                ScalarProd(N, R, R, rr);
                if (rr[0] < eps * eps)
                {
                    break;
                }

                float beta = rr[0] / r[0];
                Saxpy(P, R, beta, P, N);                // P = R + beta*P
                ++k;
            }
        }

        public static void Copy(float[] res, float[] src, int N)
        {
            Parallel.For(0, N, (i) =>
            {
                res[i] = src[i];
            });
        }

        public static void Saxpy(float[] res, float[] x, float alpha, float[] y, int N)
        {
            Parallel.For(0, N, (i) =>
            {
                res[i] = x[i] + alpha * y[i];
            });
        }

        public static void ScalarProd(int N, float[] a, float[] b, float[] result)
        {
            var cache = new SharedMemoryAllocator<float>().allocate(blockDim.x);
            int tid = threadIdx.x + blockDim.x * blockIdx.x;
            int cacheIndex = threadIdx.x;

            float tmp = 0;
            while (tid < N)
            {
                tmp += a[tid] * b[tid];
                tid += blockDim.x * gridDim.x;
            }

            cache[cacheIndex] = tmp;

            CUDAIntrinsics.__syncthreads();

            int i = blockDim.x / 2;
            while (i != 0)
            {
                if (cacheIndex < i)
                {
                    cache[cacheIndex] += cache[cacheIndex + i];
                }

                CUDAIntrinsics.__syncthreads();
                i >>= 1;
            }

            if (cacheIndex == 0)
            {
                AtomicAdd(ref result[0], cache[0]);
            }
        }

        [IntrinsicFunction("atomicAdd")]
        public static float AtomicAdd(ref float location1, float value)
        {
            float newCurrentValue = location1; // non-volatile read, so may be stale
            while (true)
            {
                float currentValue = newCurrentValue;
                float newValue = currentValue + value;
                newCurrentValue = Interlocked.CompareExchange(ref location1, newValue, currentValue);
                if (newCurrentValue == currentValue)
                    return newValue;
            }
        }

        public static double Add(ref double location1, double value)
        {
            double newCurrentValue = location1; // non-volatile read, so may be stale
            while (true)
            {
                double currentValue = newCurrentValue;
                double newValue = currentValue + value;
                newCurrentValue = Interlocked.CompareExchange(ref location1, newValue, currentValue);
                if (newCurrentValue == currentValue)
                    return newValue;
            }
        }

        // res = A - m*v
        public static void Fmsub(float[] res, float[] A, SparseMatrix m, float[] v, int N)
        {
            Parallel.For(0, N, (i) =>
            {
                int rowless = m.rows[i];
                int rowup = m.rows[i + 1];
                float tmp = A[i];
                for (int j = rowless; j < rowup; ++j)
                {
                    tmp -= v[m.indices[j]] * m.data[j];
                }

                res[i] = tmp;
            });
        }

        public static void Multiply(float[] res, SparseMatrix m, float[] v, int N)
        {
            Parallel.For(0, N, (i) =>
            {
                int rowless = m.rows[i];
                int rowup = m.rows[i + 1];
                float tmp = 0.0F;
                for (int j = rowless; j < rowup; ++j)
                {
                    tmp += v[m.indices[j]] * m.data[j];
                }

                res[i] = tmp;
            });
        }
    }
}
