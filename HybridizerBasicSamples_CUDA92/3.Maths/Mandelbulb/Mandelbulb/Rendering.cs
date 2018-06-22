using Hybridizer.Runtime.CUDAImports;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mandelbulb
{
    public class Rendering
    {
        public static float lightAngle = 140.0F;
        public static float viewAngle = 150.0F;
        public static float eyeDistanceFromNearField = 2.2F;

        public static void BitMap(string fileName, int width, int height)
        {
            Bitmap image = new Bitmap(width, height, PixelFormat.Format32bppRgb);

            float depth_of_field = 2.5F;

            float rad = MathFunctions.toRad(lightAngle);
            var lightX = MathFunctions.cosf(rad) * depth_of_field / 2;
            var lightZ = MathFunctions.sinf(rad) * depth_of_field / 2;

            float3 lightLocation = new float3(lightX, depth_of_field / 2, lightZ);
            float3 lightDirection = new float3(0.0F, 0.0F, 0.0F) - lightLocation;
            MathFunctions.normalize(ref lightDirection);

            float viewRad = MathFunctions.toRad(viewAngle);
            float viewX = MathFunctions.cosf(viewRad) * depth_of_field / 2;
            float viewZ = MathFunctions.sinf(viewRad) * depth_of_field / 2;

            float3 nearFieldLocation = new float3(viewX, 0.0F, viewZ);
            float3 viewDirection = new float3(0.0F, 0.0F, 0.0F) - nearFieldLocation;
            MathFunctions.normalize(ref viewDirection);

            float3 reverseDirection = viewDirection * eyeDistanceFromNearField;
            MathFunctions.scalarMultiply(ref reverseDirection, eyeDistanceFromNearField);
            float3 eyeLocation = nearFieldLocation - reverseDirection;

            uchar4[] imageData = new uchar4[width * height];
            
            Mandelbulb mandelbulb = new Mandelbulb { DEPTH_OF_FIELD = depth_of_field, Iterations = 200, MAX_ITER = 5000 };
            

            HybRunner runner = HybRunner.Cuda().SetDistrib(32, 32, 16, 16, 1, 0);
            dynamic wrapped = runner.Wrap(mandelbulb);

            cuda.ERROR_CHECK(cuda.GetLastError());
            cuda.ERROR_CHECK(cuda.DeviceSynchronize());
            Stopwatch watch = new Stopwatch();
            watch.Start();

            wrapped.Render(imageData, width, height, viewDirection, nearFieldLocation, eyeLocation, lightDirection);

            cuda.ERROR_CHECK(cuda.GetLastError());
            cuda.ERROR_CHECK(cuda.DeviceSynchronize());
            watch.Stop();
            //Console.WriteLine("C# time : {0} ms", watch.ElapsedMilliseconds);

            for (int y = 0; y < height; ++y)
            {
                for(int x = 0; x < width; ++x)
                {
                    uchar4 color = imageData[y * width + x];
                    image.SetPixel(x, y, Color.FromArgb(color.x, color.y, color.z));
                }
            }

            image.Save(fileName);
        }
    }
}
