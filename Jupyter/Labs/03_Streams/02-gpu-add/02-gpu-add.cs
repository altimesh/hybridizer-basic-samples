using System;
using System.Threading.Tasks;
using Hybridizer.Runtime.CUDAImports;

namespace Stream
{
    public class Program
    {
        // decorate method here
        public static void Add( float[] a, float[] b, int N)
        {
            //TODO : a += b  using threadIdx and BlockIdx
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

            // add boilerplate code to invoke generated method on the GPU. 

            Add(a, b, N); //call it on the GPU

            cuda.DeviceSynchronize(); // kernel calls are asynchronous. We need to wait for it to terminate.

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