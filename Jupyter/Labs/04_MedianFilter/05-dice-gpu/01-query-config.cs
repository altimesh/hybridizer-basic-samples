using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;

using Hybridizer.Runtime.CUDAImports;

namespace MedianFilter
{
    class Program
    {
        [EntryPoint]
        static void Dummy() {}
        
        static void Main(string[] args)
        {
            int currentDevice ;
            cuda.GetDevice(out currentDevice) ;
            
            cudaDeviceProp prop;
            cuda.GetDeviceProperties(out prop, currentDevice);
            Console.Out.WriteLine("Device properties:");
            Console.Out.WriteLine("\tname : {0}", new String(prop.name)) ;
            Console.Out.WriteLine("\tmultiProcessorCount : {0}", prop.multiProcessorCount) ;
            
            int mm = prop.major * 10 + prop.minor;
            int count = 0 ;
            
            switch (mm)
            {
                case 35: count = 192; break;
                case 37: count = 192;break;
                case 50: count = 128;break;
                case 60: count = 64;break;
                case 61: count = 128;break;
                case 70: count = 64;break;
                default:count = 0 ; break;
            }
            
            Console.Out.WriteLine("\tTotal cores : {0}", count * prop.multiProcessorCount) ;
            
            HybRunner runner = HybRunner.Cuda();
            Console.Out.WriteLine("Runner configuration:");
            Console.Out.WriteLine("\tGridDimX = {0}", runner.GridDimX) ;
            Console.Out.WriteLine("\tGridDimY = {0}", runner.GridDimY) ;
            Console.Out.WriteLine("\tBlockDimX = {0}", runner.BlockDimX) ;
            Console.Out.WriteLine("\tBlockDimY = {0}", runner.BlockDimY) ;
            Console.Out.WriteLine("\tBlockDimZ = {0}", runner.BlockDimZ) ;
            
            Console.Out.WriteLine("TOTAL parallelization = {0}", runner.GridDimX * runner.GridDimY * runner.BlockDimX * runner.BlockDimY * runner.BlockDimZ) ;
        }
    }
}