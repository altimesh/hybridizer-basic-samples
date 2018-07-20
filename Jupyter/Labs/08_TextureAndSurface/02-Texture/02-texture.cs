using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Hybridizer.Runtime.CUDAImports;

namespace TextureAndSurface
{
    class Program
    {
        static void Main(string[] args)
        {
            HybRunner runner = HybRunner.Cuda().SetDistrib(32, 32, 16, 16, 1, 0);
            
            GrayBitmap image = GrayBitmap.Load("../images/lena512.bmp");
            uint height = image.Height, width = image.Width;
            ushort[] inputPixels = image.PixelsUShort;
            
            float[] imageFloat = new float[width * height];
            float[] imageCompute = new float[width * height];
            for (int i = 0; i < width * height; ++i)
            {
                imageFloat[i] = (float)inputPixels[i];
            }
            
            dynamic wrapper = runner.Wrap(new Program());

            wrapper.Sobel(imageFloat, imageCompute, (int)width, (int)height);

            ushort[] outputPixel = new ushort[width * height];
            for (int i = 0; i < width * height; ++i)
            {
                outputPixel[i] = (ushort)imageCompute[i];
            }
            
            GrayBitmap imageSobel = new GrayBitmap(width, height);
            imageSobel.PixelsUShort = outputPixel;
            imageSobel.Save("../output-02-texture/sobel.bmp");
        }

        [EntryPoint]
        public static void Sobel(float[] input, float[] output, int width, int height)
        {
            for (int i = threadIdx.y + blockIdx.y * blockDim.y; i < height; i += blockDim.y * gridDim.y)
            {
                for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < width; j += blockDim.x * gridDim.x)
                {
                    int pixelId = i * width + j;
                    if (i != 0 && j != 0 && i != height - 1 && j != width - 1)
                    {
                        float tl = input[pixelId - width - 1];
                        float t = input[pixelId - width];
                        float tr = input[pixelId - width + 1];
                        float l = input[pixelId - 1];
                        float r = input[pixelId + 1];
                        float br = input[pixelId + width + 1];
                        float bl = input[pixelId + width - 1];
                        float b = input[pixelId + width];

                        float data = (ushort)(Math.Abs((tl + 2.0F * l + bl - tr - 2.0F * r - br)) +
                                     Math.Abs((tl + 2.0F * t + tr - bl - 2.0F * b - br)));

                        if (data > 255)
                        {
                            data = 255;
                        }

                        output[pixelId] = data;
                    }
                    else
                    {
                        output[pixelId] = 0;
                    }
                }
            }
        }
    }
}
