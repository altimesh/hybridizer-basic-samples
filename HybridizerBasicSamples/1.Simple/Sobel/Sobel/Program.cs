using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;

namespace Sobel
{
    class Program
    {

        public static int ComputeSobel(int coordX, int coordY, Bitmap baseImage)
        {
            int valPixel = 0, hor = 0 , vert = 0;

            /*
             *      Memorize the convolution of the horizontal sobel filter 
             *      1  0  -1
             *      2  0  -2
             *      1  0  -1
             */
            hor = baseImage.GetPixel(coordX + 1, coordY -1).B +
                    2 * baseImage.GetPixel(coordX + 1, coordY).B +
                    baseImage.GetPixel(coordX + 1, coordY + 1).B -
                    baseImage.GetPixel(coordX - 1, coordY - 1).B -
                    2 * baseImage.GetPixel(coordX - 1, coordY).B -
                    baseImage.GetPixel(coordX - 1, coordY + 1).B;

            /*
             *      Memorize the convolution of the vertical sobel filter 
             *       1   2   1
             *       0   0   0
             *      -1  -2  -1
             */
            vert = baseImage.GetPixel(coordX + 1, coordY + 1).B +
                    2 * baseImage.GetPixel(coordX, coordY + 1).B +
                    baseImage.GetPixel(coordX - 1, coordY + 1).B -
                    baseImage.GetPixel(coordX + 1, coordY - 1).B -
                    2 * baseImage.GetPixel(coordX, coordY - 1).B -
                    baseImage.GetPixel(coordX - 1, coordY - 1).B;
            
            //calculate the value in shade of grey of the pixel
            valPixel = Math.Abs(hor) + Math.Abs(vert);

            //test if the result is not between 0 and 255
            if (valPixel < 0) valPixel = 0;
            else if (valPixel > 255) valPixel = 255;          

            return valPixel;
        }

        public static void ComputeSobel(ref byte[] outputPixel, byte[] inputPixel, int width, int height)
        {

            
        }

        public static void ReadImage(ref byte[] inputPixel, Bitmap image, int width, int height)
        {
            for(int i = 0; i < height; ++i)
            {
                for(int j = 0; j < width; ++j)
                {
                    inputPixel[i*height + j] = image.GetPixel(i, j).B;
                }
            }
        }

        public static void SaveImage(string nameImage, byte[] outputPixel, int width, int height)
        {
            Bitmap resImage = new Bitmap(width, height);
            int col = 0;
            for (int i = 0; i < height; ++i)
            {
                for (int j = 0; j < width; ++j)
                {
                    col = outputPixel[i * height + j];
                    resImage.SetPixel(i, j, Color.FromArgb(col, col, col));
                }
            }

            //store the result image.
            resImage.Save(nameImage, System.Drawing.Imaging.ImageFormat.Png);
        }

        static void Main(string[] args)
        {
            //load the base image
            Bitmap baseImage = (Bitmap)Image.FromFile("lena512.bmp");

            //Memorize the height and the width of the base image
            int height = baseImage.Height, width = baseImage.Width, grey;

            //create an Bitmap with same dimension than base Image
            Bitmap resImage = new Bitmap(width, height);

            byte[] inputPixels = new byte[width * height];
            byte[] outputPixels = new byte[width * height];

            ReadImage(ref inputPixels, baseImage, width, height);

            ComputeSobel(ref outputPixels, inputPixels, width, height);

            SaveImage("lena-sobel.bmp", outputPixels, width, height);
        }
    }
}
