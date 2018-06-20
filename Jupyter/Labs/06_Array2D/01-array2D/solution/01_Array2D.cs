using Hybridizer.Runtime.CUDAImports;
using System.Diagnostics;
using System.Threading.Tasks;
using System;

namespace Hybridizer.Basic.Imaging
{
    class Program
    {
        static void Main(string[] args)
        {
            const int size = 512;       
            byte[,] input = new byte[size, size];
            for(int i = 0; i < size ; ++i)
            {
                for(int j =0; j < size; ++j)
                {
                    input[i,j] = (byte) 1;
                }
            }

            dynamic wrapper = HybRunner.Cuda().SetDistrib(32,32,16,16,1,0).Wrap(new Program());

            wrapper.Run(input);
            
            for(int i = 0; i < size; ++i)
            {
                for(int j = 0; j < size; ++j)
                {
                    if(input[i,j] != (byte)((i+j)%256))
                        Console.Out.WriteLine("error in " + i + " " + j);
                }
            }
            
            Console.Out.WriteLine("DONE");
            
            
        }


        [EntryPoint]
        public static void Run(byte[,] data)
        {
            int size = data.GetLength(0);
            for (int i = threadIdx.y + blockIdx.y * blockDim.y; i < size; i += blockDim.y * gridDim.y)
            {
                for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < size; j += blockDim.x * gridDim.x)
                {
                    data[i,j] = (byte)((i+j)%256);
                }
            }
        }
    }
}