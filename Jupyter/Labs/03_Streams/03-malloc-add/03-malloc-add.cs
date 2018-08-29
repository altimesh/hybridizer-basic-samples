using Hybridizer.Runtime.CUDAImports;
using System;
using System.Threading.Tasks;
using System.Runtime.InteropServices;

namespace Streams
{
    public class Program
    {
        [EntryPoint]
        public static void Add(float[] a, float[] b, int N)
        {
            Parallel.For(0, N, i =>
            {
                a[i] += b[i];
            });
        }
        
        //add unsafe
        static void Main(string[] args)
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

            for (int i = 0; i < N; ++i)
            {
                if (a[i] != (float)i + 1.0F)
                {
                    Console.Error.WriteLine("ERROR at {0} -- {1} != {2}", i, a[i], i + 1);
                    Environment.Exit(6); // abort
                }
            }

            //free the GCHandle
        }
    }
}