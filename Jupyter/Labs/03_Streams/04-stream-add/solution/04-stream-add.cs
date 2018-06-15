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
                a[i] += b[i];
            });
        }

        unsafe static void Main(string[] args)
        {
            int nStreams = 8;
            cudaStream_t[] streams = new cudaStream_t[nStreams];
            for (int k = 0; k < nStreams; ++k)
            {
                cuda.StreamCreate(out streams[k]);
            }

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

            int slice = N / nStreams;

            dynamic wrapped = HybRunner.Cuda().Wrap(new Program());

            for (int k = 0; k < nStreams; ++k)
            {
                int start = k * slice;
                int stop = start + slice;
                wrapped.SetStream(streams[k]).Add(d_a, d_b, start, stop);
            }

            for (int k = 0; k < nStreams; ++k)
            {
                int start = k * slice;
                cuda.MemcpyAsync(h_a + start * sizeof(float), d_a + start * sizeof(float), slice * sizeof(float), cudaMemcpyKind.cudaMemcpyDeviceToHost, streams[k]);
            }

            for (int k = 0; k < nStreams; ++k)
            {
                cuda.StreamSynchronize(streams[k]);
                cuda.StreamDestroy(streams[k]);
            }

            for (int i = 0; i < N; ++i)
            {
                if (a[i] != (float)i + 1.0F)
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