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

        static void Main(string[] args)
        {
            //load the base image
            Bitmap baseImage = (Bitmap)Image.FromFile("lena512.bmp");

            //Memorize the height and the width of the base image
            int height = baseImage.Height, width = baseImage.Width, grey;

            //create an Bitmap with same dimension than base Image
            Bitmap resImage = new Bitmap(width, height);

            for (int i = 0; i < height; ++i)
                for(int j = 0 ; j < width; ++j)
                {
                    //Don't compute the borders, then let the pixel in black.
                    if(i == 0 || j == 0 || i == height -1 || j == width -1) resImage.SetPixel(i, j, Color.FromArgb(0, 0, 0));
                    else
                    {
                        //Do the sobel filter on the i,j coordonate and memorize the result
                        grey = ComputeSobel(i, j, baseImage);

                        //Affect the result on the pixel of the result image; 
                        resImage.SetPixel(i, j, Color.FromArgb(grey, grey, grey));
                    }
                }
            //store the result image.
            resImage.Save("sobel.png", System.Drawing.Imaging.ImageFormat.Png);
        }
    }
}
