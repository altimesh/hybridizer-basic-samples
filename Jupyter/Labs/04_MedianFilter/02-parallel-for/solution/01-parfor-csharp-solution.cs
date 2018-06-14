using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;

using Hybridizer.Imaging;

namespace MedianFilter
{
    class Program
    {

        public static void NaiveCsharp(ushort[] output, ushort[] input, int width, int height, int window)
        {
            int windowCount = 2 * window + 1;
            Parallel.For(window, height - window, j =>
            {
                var buffer = new ushort[windowCount * windowCount];
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

                    Array.Sort(buffer, 0, windowCount * windowCount);
                    output[j * width + i] = buffer[(windowCount * windowCount) / 2];
                }
            }) ;
        }
        
        static void Main(string[] args)
        {
            GrayBitmap image = GrayBitmap.Load("../../../images/lena_highres_greyscale_noise.bmp");
            GrayBitmap denoised = new GrayBitmap(image.Width, image.Height) ;
            ushort[] input = image.PixelsUShort ;
            ushort[] output = new ushort[image.Width * image.Height];
            
            int window = 3 ;
            
            Stopwatch watch = new Stopwatch();
            watch.Start();
            
            NaiveCsharp (output, input, (int)image.Width, (int)image.Height, window) ;
            
            watch.Stop();
            string time = String.Format("{0:0.00}", watch.ElapsedMilliseconds * 1.0E-3);

            Console.WriteLine ($"Parallel.For time : {time}");
            denoised.PixelsUShort = output ;
            denoised.Save ("../../../output-02-parfor/denoised.bmp");
        }
    }
}
        