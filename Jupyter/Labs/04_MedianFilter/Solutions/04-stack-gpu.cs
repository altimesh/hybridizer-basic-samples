using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;

using Hybridizer.Imaging;
using Hybridizer.Runtime.CUDAImports;

namespace MedianFilter
{
    public class Filter
    {
        const int window = 5 ;
        const int windowCount = window + window + 1 ;
            
        [IntrinsicInclude("bitonicsort.cuh")]
        public class Bitonic
        {
            [IntrinsicFunction("bitonicsort<unsigned short,int>")]
            public static void Sort(ushort[] data, int from, int to)
            {
                Array.Sort(data, from, to - from);
            }
        }
        
        [EntryPoint]
        public static void ParForGPU(ushort[] output, ushort[] input, int width, int height)
        {
            Parallel.For(window, height - window, j =>
            {
                // var buffer = new ushort[windowCount * windowCount];
                var buffer = new StackArray<ushort>(windowCount * windowCount);
                for (int i = window; i < width - window; ++i)
                {
                    for (int k = -window; k <= window; ++k)
                    {
                        for (int p = -window; p <= window; ++p)
                        {
                            int bufferIndex = (k + window) * windowCount + p + window;
                            int pixelIndex = (j + k) * width + (i + p);
                            buffer[bufferIndex] = input[pixelIndex];
                        }
                    }

                    Bitonic.Sort(buffer.data, 0, windowCount * windowCount);
                    output[j * width + i] = buffer[(windowCount * windowCount) / 2];
                }
            }) ;
        }
    }
    
    class Program
    {
        
        static void Main(string[] args)
        {
            GrayBitmap image = GrayBitmap.Load("../../../images/lena_highres_greyscale_noise.bmp");
            GrayBitmap denoised = new GrayBitmap(image.Width, image.Height) ;
            ushort[] input = image.PixelsUShort ;
            ushort[] output = new ushort[image.Width * image.Height];
            
            Stopwatch watch = new Stopwatch();
            watch.Start();
            
            int window = 3 ;
            
            // create an instance of runner
            HybRunner runner = HybRunner.Cuda();
            // wrap a new instance of Program
            dynamic wrapper = runner.Wrap(new Filter());
            // run the method on GPU
            wrapper.ParForGPU (output, input, (int)image.Width, (int)image.Height) ;
                        
            watch.Stop();
            string time = String.Format("{0:0.00}", watch.ElapsedMilliseconds * 1.0E-3);

            Console.WriteLine ($"StackArray GPU time : {time}");
            denoised.PixelsUShort = output ;
            denoised.Save ("../../../output-04-stack-gpu/denoised.bmp");
        }
    }
}
        