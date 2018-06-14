using System;
using System.Threading.Tasks;
using Hybridizer.Runtime.CUDAImports;

namespace Stream
{
    public class Program
    {
        [EntryPoint]
        public static void Add( float[] a, float[] b, int N)
        {
            for (int k = threadIdx.x + blockDim.x * blockIdx.x; k < N; k += blockDim.x * gridDim.x)
            {
                a[k] += b[k];
            }
        }

        public static void Main()
        {
            const int N = 1024 * 1024 * 32;
            float[] a = new float[N];
            float[] b = new float[N];

            for (int i = 0; i < N; ++i)
            {
                a[i] = (float)i;
                b[i] = 1.0F;
            }

            dynamic wrapped = HybRunner.Cuda().Wrap(new Program());

            wrapped.Add(a, b, N);

            cuda.DeviceSynchronize(); 

            for (int i = 0; i < N; ++i)
            {
                if (a[i] != (float)i + 1.0F)
                {
                    Console.Error.WriteLine("ERROR at {0} -- {1} != {2}", i, a[i], i + 1);
                    Environment.Exit(6); // abort
                }
            }

            Console.Out.WriteLine("OK");
        }
    }
}