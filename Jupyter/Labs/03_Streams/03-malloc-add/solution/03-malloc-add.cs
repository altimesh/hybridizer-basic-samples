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

            IntPtr d_a, d_b; //device pointer
            cuda.Malloc(out d_a, N * sizeof(float));
            cuda.Malloc(out d_b, N * sizeof(float));

            GCHandle handle_a = GCHandle.Alloc(a, GCHandleType.Pinned);
            GCHandle handle_b = GCHandle.Alloc(b, GCHandleType.Pinned);
            IntPtr h_a = handle_a.AddrOfPinnedObject();
            IntPtr h_b = handle_b.AddrOfPinnedObject();

            cuda.DeviceSynchronize();

            cuda.Memcpy(d_a, h_a, N * sizeof(float), cudaMemcpyKind.cudaMemcpyHostToDevice);
            cuda.Memcpy(d_b, h_b, N * sizeof(float), cudaMemcpyKind.cudaMemcpyHostToDevice);

            dynamic wrapped = HybRunner.Cuda().Wrap(new Program());

            wrapped.Add(d_a, d_b, N);

            cuda.DeviceSynchronize();

            cuda.Memcpy(h_a, d_a , N * sizeof(float), cudaMemcpyKind.cudaMemcpyDeviceToHost);

            for (int k = 0; k < 10; ++k)
            {
                Console.WriteLine(a[k]);
            }

            handle_a.Free();
            handle_b.Free();
        }
    }
}