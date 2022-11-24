using System.Drawing;
using Hybridizer.Runtime.CUDAImports;
using System.Diagnostics;
using System;

namespace Hybridizer.Basic.Imaging
{
    class Program
    {
        static void Main(string[] args)
        {
            Bitmap baseImage = (Bitmap)Image.FromFile("lena512.bmp");
            const int size = 512;

            Bitmap resImage = new Bitmap(size, size);

            byte[,] inputPixels = new byte[size, size];
            byte[,] outputPixels = new byte[size, size];

            ReadImage(inputPixels, baseImage, size);

            HybRunner runner = HybRunner.Cuda().SetDistrib(32, 32, 16, 16, 1, 0);
            dynamic wrapper = runner.Wrap(new Program());

            wrapper.ComputeSobel(outputPixels, inputPixels);

            SaveImage("lena-sobel.bmp", outputPixels, size);
			try { Process.Start("lena-sobel.bmp");} catch {} // catch exception for non interactives machines
        }

        public static void ReadImage(byte[,] inputPixel, Bitmap image, int size)
        {
            for (int i = 0; i < size; ++i)
            {
                for (int j = 0; j < size; ++j)
                {
                    double greyPixel = (image.GetPixel(i, j).R * 0.2126 + image.GetPixel(i, j).G * 0.7152 + image.GetPixel(i, j).B * 0.0722);
                    inputPixel[i, j] = Convert.ToByte(greyPixel);
                }
            }
        }

        [EntryPoint]
        public static void ComputeSobel(byte[,] outputPixel, byte[,] inputPixel)
        {
            int size = inputPixel.GetLength(0);
            for (int i = threadIdx.y + blockIdx.y * blockDim.y; i < size; i += blockDim.y * gridDim.y)
            {
                for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < size; j += blockDim.x * gridDim.x)
                {   
                    int output = 0;
                    if (i > 0 && j > 0 && i < size - 1 && j < size - 1)
                    {
                        byte topl = inputPixel[i - 1, j - 1];
                        byte top = inputPixel[i - 1, j];
                        byte topr = inputPixel[i - 1, j + 1];
                        byte l = inputPixel[i, j - 1];
                        byte r = inputPixel[i, j + 1];
                        byte botl = inputPixel[i + 1, j - 1];
                        byte bot = inputPixel[i + 1, j];
                        byte botr = inputPixel[i + 1, j + 1];

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

                        outputPixel[i, j] = (byte)output;
                    }
                }
            }
        }

        public static void SaveImage(string nameImage, byte[,] outputPixel, int size)
        {
            Bitmap resImage = new Bitmap(size, size);
            int col = 0;
            for (int i = 0; i < size; ++i)
            {
                for (int j = 0; j < size; ++j)
                {
                    col = outputPixel[i, j];
                    resImage.SetPixel(i, j, Color.FromArgb(col, col, col));
                }
            }

            //store the result image.
            resImage.Save(nameImage, System.Drawing.Imaging.ImageFormat.Png);
        }

    }
}
