using Hybridizer.Runtime.CUDAImports;
using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace Streams
{
    public class Program
    {
        [EntryPoint]
        public static void Add(float[] a, float[] b, int start, int stop)
        {
            Parallel.For(start, stop, i =>
            {
                for(int i = 0; i < 100; ++i)
                    a[i] += b[i];
            });
        }

        unsafe static void Main(string[] args)
        {
            int nStreams = 8;
            cudaStream_t[] streams = new cudaStream_t[nStreams];
            
            //create streams

            int N = 1024 * 1024 * 32;
            float[] a = new float[N];
            float[] b = new float[N];

            for (int k = 0; k < N; ++k)
            {
                a[k] = (float)k;
                b[k] = 1.0F;
            }

            IntPtr d_a, d_b; // device pointers
            cuda.Malloc(out d_a, N * sizeof(float));
            cuda.Malloc(out d_b, N * sizeof(float));

            GCHandle handle_a = GCHandle.Alloc(a, GCHandleType.Pinned);
            GCHandle handle_b = GCHandle.Alloc(b, GCHandleType.Pinned);
            IntPtr h_a = handle_a.AddrOfPinnedObject();
            IntPtr h_b = handle_b.AddrOfPinnedObject();

            cuda.DeviceSynchronize();

            cuda.Memcpy(d_a, h_a, N * sizeof(float), cudaMemcpyKind.cudaMemcpyHostToDevice);
            cuda.Memcpy(d_b, h_b, N * sizeof(float), cudaMemcpyKind.cudaMemcpyHostToDevice);

            int slice = N / nStreams; // size of the array compute by each stream 

            dynamic wrapped = HybRunner.Cuda().Wrap(new Program());
            int start;
            int stop;
            
            // call kernel with each stream 

            // copy data device to host

            // synchronize and destroy streams

            for (int i = 0; i < N; ++i)
            {
                if (a[i] != (float)i + (1.0F * 100.0F))
                {
                    Console.Error.WriteLine("ERROR at {0} -- {1} != {2}", i, a[i], i + 1);
                    Environment.Exit(6); // abort
                }
            }

            handle_a.Free();
            handle_b.Free();
        }
    }
}