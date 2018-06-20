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
            byte[] input = new byte[size * size]; //change it to a 2D array
            for(int i = 0; i < size ; ++i)
            {
                input[i] = (byte) 1;
            }

            dynamic wrapper = HybRunner.Cuda().Wrap(new Program()); //add the SetDistrib function

            wrapper.Run(input, size);
            
            for(int i = 0; i < size; ++i)
            {  
                if(input[i] != (byte)(i%256))
                    Console.Out.WriteLine("error in " + i ); 
            }
            
            Console.Out.WriteLine("DONE");
        }


        [EntryPoint]
        public static void Run(byte[] data, int size)
        {         
            for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size * size; i += blockDim.x * gridDim.x)
            {
                data[i] = (byte)(i%256);
            }
            
        }
    }
}