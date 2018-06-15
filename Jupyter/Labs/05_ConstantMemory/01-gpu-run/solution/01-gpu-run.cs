using Hybridizer.Runtime.CUDAImports;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConstantMemory
{
    class Program
    {
        [EntryPoint]
        public static void Run(float[] output, float[] input, float[] data, int N) 
        {
            for(int k = threadIdx.x + blockDim.x * blockIdx.x; k < N; k += blockDim.x * gridDim.x)
            {
                float tmp = 0;
                for(int p = -2; p <= 2; ++p)
                {
                    tmp += data[p + 2] * input[k];
                }

                output[k] = tmp;
            }
        }

        static void Main(string[] args)
        {
            const int N = 1024 * 1024 * 32;
            float[] data = { -2.0F, -1.0F, 0.0F, 1.0F, 2.0F };
            float[] input = new float[N];
            float[] output = new float[N];
            Random rand = new Random();
            for(int k = 0; k < N; ++k)
            {
                output[k] = (float)rand.NextDouble();
            }

            dynamic wrapped = HybRunner.Cuda().Wrap(new Program());

            wrapped.Run(input, output, data, N);

            Console.Out.WriteLine("DONE");
        }
    }
}