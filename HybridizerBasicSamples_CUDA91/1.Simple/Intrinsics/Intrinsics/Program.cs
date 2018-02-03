using Hybridizer.Runtime.CUDAImports;
using System;
using System.Threading.Tasks;

namespace Intrinsics
{
    class Program
    {
        public static half2 getHalf2(float x)
        {
            return new half2(x, x);
        }

        public static half2 exp(half2 x)
        {
            return ((((((((((((((getHalf2(15.0F) + x)
                * x + getHalf2(210.0F))
                * x + getHalf2(2730.0F))
                * x + getHalf2(32760.0F))
                * x + getHalf2(360360.0F))
                * x + getHalf2(3603600.0F))
                * x + getHalf2(32432400.0F))
                * x + getHalf2(259459200.0F))
                * x + getHalf2(1816214400.0F))
                * x + getHalf2(10897286400.0F))
                * x + getHalf2(54486432000.0F))
                * x + getHalf2(217945728000.0F))
                * x + getHalf2(653837184000.0F))
                * x + getHalf2(1307674368000.0F))
                * x * getHalf2(7.6471637318198164759011319857881e-13F);
        }

        [EntryPoint]
        public static void Compute(half2[] input, int N)
        {
            Parallel.For(0, N, i =>
            {
                input[i] = exp(input[i]);
            });
        }

        static void Main(string[] args)
        {
            const int N = 1024 * 32;
            half2[] input = new half2[N];

            cudaDeviceProp prop;
            cuda.GetDeviceProperties(out prop, 0);
            int computeCapability = 10 * prop.major + prop.minor;
            if(computeCapability < 53)
            {
                Console.Error.WriteLine("Your device doesn't support half precision");
                Environment.Exit(6); // abort
            }

            HybRunner runner = HybRunner.Cuda("Intrinsics_CUDA.dll").SetDistrib(16 * prop.multiProcessorCount, 128);
            dynamic wrapped = runner.Wrap(new Program());
            wrapped.Compute(input, N);
            Console.WriteLine(cuda.GetErrorString(cuda.DeviceSynchronize()));
        }
    }
}
