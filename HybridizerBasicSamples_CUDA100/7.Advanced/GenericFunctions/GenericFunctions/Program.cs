using Hybridizer.Runtime.CUDAImports;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GenericFunctions
{
    class Vector<T> where T: struct
    {
        T[] _data;
        int _length;

        public Vector(T[] data)
        {
            _data = data;
            _length = data.Length;
        }

        [Kernel]
        public int Length { get => _length; }

        [Kernel]
        public T this[int i] { get => _data[i]; set { _data[i] = value; } }
    }

    class Operations
    {
        [Kernel]
        public static void Invoke<T>(Vector<T> v, int i, T val, Func<T, T, T> func) where T: struct
        {
            v[i] = func(v[i], val);
        }

        [EntryPoint]
        public static void AddOne(Vector<float> v)
        {
            int N = v.Length;
            for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < N; i += blockDim.x * gridDim.x)
            { 
                Invoke(v, i, 1.0F, (x, y) => x + y);
            }
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            const int N = 1024 * 1024 * 32;
            float[] a = new float[N];

            HybRunner.Cuda().Wrap(new Operations()).AddOne(new Vector<float>(a));

            for(int i = 0; i < N; ++i)
            {
                if(a[i] != 1.0F)
                {
                    Console.Error.WriteLine($"ERROR at {i} : got {a[i]} instead of 1.0F");
                    Environment.Exit(6); // abort
                }
            }

            Console.Out.Write("OK");
        }
    }
}
