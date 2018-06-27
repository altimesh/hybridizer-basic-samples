using Hybridizer.Runtime.CUDAImports;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;

namespace Mandelbulb
{
    public class Rendering
    {
        public static float lightAngle = 140.0F;
        public static float viewAngle = 150.0F;
        public static float eyeDistanceFromNearField = 2.2F;
        public float depth_of_field;
        public float3 viewDirection;
        public float3 nearFieldLocation;
        public float3 eyeLocation;
        public float3 lightDirection;
        dynamic wrapped;
        Mandelbulb mandelbulb;

        public Rendering(float depthoffield)
        {
            depth_of_field = depthoffield;
            float rad = MathFunctions.toRad(lightAngle);
            var lightX = MathFunctions.cosf(rad) * depth_of_field / 2;
            var lightZ = MathFunctions.sinf(rad) * depth_of_field / 2;

            float3 lightLocation = new float3(lightX, depth_of_field / 2, lightZ);
            lightDirection = new float3(0.0F, 0.0F, 0.0F) - lightLocation;
            MathFunctions.normalize(ref lightDirection);

            float viewRad = MathFunctions.toRad(viewAngle);
            float viewX = MathFunctions.cosf(viewRad) * depth_of_field / 2;
            float viewZ = MathFunctions.sinf(viewRad) * depth_of_field / 2;

            nearFieldLocation = new float3(viewX, 0.0F, viewZ);
            viewDirection = new float3(0.0F, 0.0F, 0.0F) - nearFieldLocation;
            MathFunctions.normalize(ref viewDirection);

            float3 reverseDirection = viewDirection * eyeDistanceFromNearField;
            MathFunctions.scalarMultiply(ref reverseDirection, eyeDistanceFromNearField);
            eyeLocation = nearFieldLocation - reverseDirection;

            mandelbulb = new Mandelbulb { DEPTH_OF_FIELD = depth_of_field, Iterations = 200, MAX_ITER = 5000 };

            HybRunner runner = HybRunner.Cuda().SetDistrib(32, 32, 16, 16, 1, 0);
            wrapped = runner.Wrap(mandelbulb);
            cuda.ERROR_CHECK(cuda.GetLastError());
            cuda.ERROR_CHECK(cuda.DeviceSynchronize());
        }

        public void Texture(cudaSurfaceObject_t surface, int width, int height)
        {
            wrapped.Render(surface, width, height, viewDirection, nearFieldLocation, eyeLocation, lightDirection);
        }

        public void BitMap(string fileName, int width, int height)
        {
            Bitmap image = new Bitmap(width, height, PixelFormat.Format32bppRgb);
            
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
                for (int x = 0; x < width; ++x)
                {
                    uchar4 color = imageData[y * width + x];
                    image.SetPixel(x, y, Color.FromArgb(color.x, color.y, color.z));
                }
            }

            image.Save(fileName);
        }
    }
}
