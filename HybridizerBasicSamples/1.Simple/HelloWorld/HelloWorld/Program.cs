using Hybridizer.Runtime.CUDAImports;
using System;
using System.Threading.Tasks;

namespace HelloWorld
{
    class Program
    {
        [EntryPoint("run")]
        public static void Run(int N, int[] a, int[] b)
        {
            Parallel.For(0, N, i => { a[i] += b[i]; });
        }

        static void Main(string[] args)
        {
            int[] a = { 1, 2, 3, 4, 5 };
            int[] b = { 10, 20, 30, 40, 50 };

            // create an instance of HybRunner object to wrap calls on GPU
            HybRunner runner = HybRunner.Cuda("HelloWorld_CUDA.dll");

            // create a wrapper object to call GPU methods instead of C#
            dynamic wrapped = runner.Wrap(new Program());

            // run the method on GPU
            wrapped.Run(a.Length, a, b);

            // verify the results
            for (int k = 0; k < a.Length; ++k)
            {
                if (a[k] != (11 * (k+1)))
                {
                    Console.Out.WriteLine("ERROR !");
                }
            }
            Console.Out.WriteLine("DONE");
        }
    }
}
