using Hybridizer.Runtime.CUDAImports;
using System;
using System.Diagnostics;
using System.Threading.Tasks;

namespace HelloWorld
{
    class Program
    {
        [EntryPoint]
        public static void Run()
        {
            Console.Out.WriteLine("Hello World") ;
        }

        static void Main(string[] args)
        {
            dynamic wrapped = HybRunner.Cuda().SetDistrib(1,1).Wrap(new Program()) ;
            wrapped.Run () ;

            Console.Out.WriteLine("DONE");
        }
    }
}
