using Hybridizer.Runtime.CUDAImports;
using System;
using System.Diagnostics;
using System.Threading.Tasks;

namespace HelloWorld
{
    class Program
    {
        [EntryPoint("run")]
        public static void Run(int N, double[] a, double[] b)
        {
            Parallel.For(0, N, i => { a[i] += b[i]; });

            toto
        }

        static void Main(string[] args)
        {
            //Initialize a stopwatch
            Stopwatch timer = new Stopwatch();

            int N = 1024 * 1024 * 64;
            double[] acuda = new double[N];
            double[] adotnet = new double[N];

            double[] b = new double[N];

            Random rand = new Random();

            //Initialize acuda et adotnet and b by some doubles randoms, acuda and adotnet have same numbers. 
            for(int i = 0; i < N; ++i)
            {
                acuda[i] = rand.NextDouble();
                adotnet[i] = acuda[i];
                b[i] = rand.NextDouble();
            }


            // create an instance of HybRunner object to wrap calls on GPU
            HybRunner runner = HybRunner.Cuda("HelloWorld_CUDA.dll").SetDistrib(20,256);

            // create a wrapper object to call GPU methods instead of C#
            dynamic wrapped = runner.Wrap(new Program());

            //Start the stopwatch
            timer.Start();

            // run the method on GPU
            wrapped.Run(N, acuda, b);

            //Stop the stopwatch
            timer.Stop();            

            //Restart the stopwatch
            timer.Restart();

            // run the method on CPU
            Run(N, adotnet, b);

            //Stop the stopwatch
            timer.Stop();
            
            // verify the results
            for (int k = 0; k < N; ++k)
            {
                if (acuda[k] != adotnet[k])
                    Console.Out.WriteLine("ERROR !");
            }
            Console.Out.WriteLine("DONE");
        }
    }
}
