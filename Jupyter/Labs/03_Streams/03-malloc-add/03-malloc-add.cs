using Hybridizer.Runtime.CUDAImports;
using System;
using System.Runtime.InteropServices;

namespace Streams
{
    public class Program
    {
        [EntryPoint]
        public static void Add(float[] a, float[] b, int N)
        {
            for (int k = threadIdx.x + blockDim.x * blockIdx.x; k < N; k += blockDim.x * gridDim.x)
            {
                a[k] += b[k];
            }
        }

        unsafe static void Main(string[] args)
        {
            int N = 1024 * 1024 * 32;
            float[] a = new float[N];
            float[] b = new float[N];

            for (int k = 0; k < N; ++k)
            {
                a[k] = (float)k;
                b[k] = 1.0F;
            }

            // create and allocate your device pointer

            //pin your c# arrays

            //copy the data from host to device

            //add boilerplate code to execute kernel

            //copy the data from device to host

            for (int k = 0; k < 10; ++k)
            {
                Console.WriteLine(a[k]);
            }

            //free the GCHandle
        }
    }
}