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
        const int window = 3 ;
        const int windowCount = window + window + 1 ;
            
        
        public class StaticSort
        {
            [IntrinsicFunction("::hybridizer::StaticSort<49>::sort<uint16_t>")]
            public static void Sort(ushort[] data)
            {
                Array.Sort(data);
            }
        }
        
        [EntryPoint]
        public static void ParForGPU(ushort[] output, ushort[] input, int width, int height)
        {
            Parallel2D.For(window, width - window, window, height - window, (i,j) =>
            {
                var buffer = new StackArray<ushort>(windowCount * windowCount);
                for (int k = -window; k <= window; ++k)
                {
                    for (int p = -window; p <= window; ++p)
                    {
                        int bufferIndex = (k + window) * windowCount + p + window;
                        int pixelIndex = (j + k) * width + (i + p);
                        buffer[bufferIndex] = input[pixelIndex];
                    }
                }

                StaticSort.Sort(buffer.data);
                output[j * width + i] = buffer[(windowCount * windowCount) / 2];
            }) ;
        }
    }
    
    class Program
    {
        
        static void Main(string[] args)
        {
            GrayBitmap image = GrayBitmap.Load("../images/lena_highres_greyscale_noise.bmp");
            GrayBitmap denoised = new GrayBitmap(image.Width, image.Height) ;
            ushort[] input = image.PixelsUShort ;
            ushort[] output = new ushort[image.Width * image.Height];
            
            Stopwatch watch = new Stopwatch();
            watch.Start();
            
            dim3 grid = new dim3(16,16,1) ;
            dim3 block = new dim3(16,16,1) ;
            
            // create an instance of runner
            HybRunner runner = HybRunner.Cuda();
            // wrap a new instance of Program
            dynamic wrapper = runner.Wrap(new Filter());
            // run the method on GPU
            wrapper.SetDistrib(grid,block).ParForGPU (output, input, (int)image.Width, (int)image.Height) ;
                        
            watch.Stop();
            string time = String.Format("{0:0.00}", watch.ElapsedMilliseconds * 1.0E-3);
            
            string kernelTime = String.Format("{0:0.00}", runner.LastKernelDuration.ElapsedMilliseconds * 1.0E-3);

            Console.WriteLine ($"StaticSort GPU time : {time}");
            Console.WriteLine ($"StaticSort GPU -- kernel time : {kernelTime}");
            denoised.PixelsUShort = output ;
            denoised.Save ("../output-06-regsort-gpu/denoised.bmp");
        }
    }
}
        