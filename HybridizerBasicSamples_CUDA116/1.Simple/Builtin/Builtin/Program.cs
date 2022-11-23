﻿using Hybridizer.Runtime.CUDAImports;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Builtin
{
    class Program
    {
        [EntryPoint]
        public static void Run(int N, int[] a, int[] result)
        {
            Parallel.For(0, N, i => 
            {
                Interlocked.Add(ref result[0], a[i]);
            });
        }

        static void Main(string[] args)
        {
            const int N = 1024;
            int[] a = new int[N];
            for(int i = 0; i < 1024; ++i)
            {
                a[i] = i;
            }

            int[] result = new int[1];
            
            HybRunner runner = HybRunner.Cuda();

            // create a wrapper object to call GPU methods instead of C#
            dynamic wrapped = runner.Wrap(new Program());

            wrapped.Run(N, a, result);
            cuda.DeviceSynchronize();
            Console.Out.WriteLine("sum = {0}", result[0]);
        }
    }
}
