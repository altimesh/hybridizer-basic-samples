using System;

namespace VectorAdd 
{
    public class Program 
    {
        public static void Add(float[] dst, float[] a, float[] b, int N) 
        {
            for(int i = 0; i < N; ++i) 
            {
                dst[i] = a[i] + b[i];
            }
        }
    
        public static void Main() 
        {
            const int N = 1024*1024*32;
            float[] a = new float[N];
            float[] b = new float[N];
            float[] dst = new float[N];
            
            for(int i = 0; i < N; ++i) 
            {
                a[i] = (float) i;
                b[i] = 1.0F;
            }
            
            Add(dst, a, b, N);
            
            for(int i = 0; i < N; ++i) 
            {
                if(dst[i] != (float) i + 1.0F) 
                {
                    Console.Error.WriteLine("ERROR at {0} -- {1} != {2}", i, dst[i], i+1);
                    Environment.Exit(6); // abort
                }
            }
            
            Console.Out.WriteLine("OK");
        }
    }
}