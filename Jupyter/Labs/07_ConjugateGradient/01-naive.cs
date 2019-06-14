using System;
using System.Threading.Tasks;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;

namespace Hybridizer.Basic.Maths
{
    unsafe class Program
    {
        static void Main(string[] args)
        {
            int size = 1000000; // very slow convergence with no preconditioner
            SparseMatrix A = SparseMatrix.Laplacian_1D(size);
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
                Multiply(AP, A, P, N);                  // AP = A*P
                float r = ScalarProd(R, R, N);                  // save <R|R>
                float alpha = r / ScalarProd(P, AP, N);         // alpha = <R|R> / <P|AP>
                Saxpy(X, X, alpha, P, N);               // X = X - alpha*P
                Saxpy(R, R, -alpha, AP, N);             // RR = R-alpha*AP
                float rr = ScalarProd(R, R, N);
                if (rr < eps * eps)
                {
                    break;
                }

                float beta = rr / r;
                Saxpy(P, R, beta, P, N);                // P = R + beta*P
                ++k;
            }
        }
        
        public static void Copy(float[] res, float[] src, int N)
        {
            for(int i = 0; i < N; ++i)
            {
                res[i] = src[i];
            }
        }
        
        public static void Saxpy(float[] res, float[] x, float alpha, float[] y, int N)
        {
            for(int i = 0; i < N; ++i)
            {
                res[i] = x[i] + alpha * y[i];
            }
        }


        public static float ScalarProd(float[] X, float[] Y, int N)
        {
            float res = 0.0f;
            for (int i = 0; i < N; ++i)
            {
                res += X[i] * Y[i];
            }
            return res;

        }

        public static double Add(ref double location1, double value)
        {
            double newCurrentValue = location1; 
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
            for(int i = 0; i < N; ++i)
            {
                int rowless = m.rows[i];
                int rowup = m.rows[i + 1];
                float tmp = A[i];
                for (int j = rowless; j < rowup; ++j)
                {
                    tmp -= v[m.indices[j]] * m.data[j];
                }

                res[i] = tmp;
            }
        }
        
        public static void Multiply(float[] res, SparseMatrix m, float[] v, int N)
        {
            for(int i = 0; i < N; ++i)
            {
                int rowless = m.rows[i];
                int rowup = m.rows[i + 1];
                float tmp = 0.0F;
                for (int j = rowless; j < rowup; ++j)
                {
                    tmp += v[m.indices[j]] * m.data[j];
                }

                res[i] = tmp;
            }
        }
    }
}
