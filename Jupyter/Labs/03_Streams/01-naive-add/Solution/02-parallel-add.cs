using System;
using System.Threading.Tasks;

namespace Stream
{
    public class Program
    {
        public static void Add(float[] a, float[] b, int N)
        {
            Parallel.For(0, N, i =>
            {
                a[i] += b[i];
            });
        }

        public static void Main()
        {
            const int N = 1024 * 1024 * 32;
            float[] a = new float[N];
            float[] b = new float[N];

            for (int i = 0; i < N; ++i)
            {
                a[i] = (float)i;
                b[i] = 1.0F;
            }

            Add(a, b, N);

            for (int i = 0; i < N; ++i)
            {
                if (a[i] != (float)i + 1.0F)
                {
                    Console.Error.WriteLine("ERROR at {0} -- {1} != {2}", i, a[i], i + 1);
                    Environment.Exit(6); // abort
                }
            }

            Console.Out.WriteLine("OK");
        }
    }
}