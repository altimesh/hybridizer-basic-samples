using System.Drawing;
using Hybridizer.Runtime.CUDAImports;
using System.Diagnostics;
using System.Drawing.Imaging;
using System;

namespace Hybridizer.Basic.Imaging
{
    class Program
    {
        static void Main(string[] args)
        {
            //open the input image and lock the image
            Bitmap baseImage = (Bitmap)Image.FromFile("../../../images/lena_highres_greyscale.bmp");
            int height = baseImage.Height, width = baseImage.Width;
            
            //create result image and lock 
            Bitmap resImage = new Bitmap(width, height);
            
            //take pointer from locked memory
            
            byte[] inputPixels = new byte[width * height];
            byte[] outputPixels = new byte[width * height];
                
            ReadImage(inputPixels, baseImage, width, height);
            
            // pin images memory for cuda
            
            HybRunner runner = HybRunner.Cuda().SetDistrib(32, 32, 16, 16, 1, 0);
            dynamic wrapper = runner.Wrap(new Program());

            wrapper.ComputeSobel(outputPixels, inputPixels, width, height);
           
            // unregister pinned memory and unlock images
            SaveImage("lena_highres_sobel.bmp", outputPixels, width, height);
			try { Process.Start("lena_highres_sobel.bmp");} catch {} // catch exception for non interactives machines
        }
        
        //make unsafe method and change array to byte pointer
        [EntryPoint]
        public static void ComputeSobel(byte[] outputPixel, byte[] inputPixel, int width, int height)
        {
            for (int i = threadIdx.y + blockIdx.y * blockDim.y; i < height; i += blockDim.y * gridDim.y)
            {
                for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < width; j += blockDim.x * gridDim.x)
                {
                    int output = 0;
                    if (i != 0 && j != 0 && i != height - 1 && j != width - 1)
                    {
                        int pixelId = i * width + j;
                        byte topl = inputPixel[pixelId - width - 1];
                        byte top = inputPixel[pixelId - width];
                        byte topr = inputPixel[pixelId - width + 1];
                        byte l = inputPixel[pixelId - 1];
                        byte r = inputPixel[pixelId + 1];
                        byte botl = inputPixel[pixelId + width - 1];
                        byte bot = inputPixel[pixelId + width];
                        byte botr = inputPixel[pixelId + width + 1];

                        int sobelx = (topl) + (2 * l) + (botl) - (topr) - (2 * r) - (botr);
                        int sobely = (topl + 2 * top + topr - botl - 2 * bot - botr);

                        int squareSobelx = sobelx * sobelx;
                        int squareSobely = sobely * sobely;

                        output = (int)Math.Sqrt((squareSobelx + squareSobely));

                        if (output < 0)
                        {
                            output = -output;
                        }
                        if (output > 255)
                        {
                            output = 255;
                        }

                        outputPixel[pixelId] = (byte)output;
                    }
                }
            }
        }
    }
}