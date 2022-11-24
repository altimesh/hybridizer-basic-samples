using Hybridizer.Runtime.CUDAImports;
using System;

namespace MandelbrotRenderer.Mandelbrots
{
    public class Mandelbrot
    {
        // this will generate a vectorized function in AVX*, or a __device__ method in CUDA
        [Kernel]
        public static int IterCount(float cx, float cy, int maxiter)
        {
            int result = 0;
            float x = 0.0F;
            float y = 0.0F;
            float xx = 0.0F, yy = 0.0F;
            while (xx + yy <= 4.0F && result < maxiter)
            {
                xx = x * x;
                yy = y * y;
                float xtmp = xx - yy + cx;
                y = 2.0F * x * y + cy;
                x = xtmp;
                result += 1;
            }

            return result;
        }

        [EntryPoint]
        public static void Render(int[] output, 
                                         float fx, float fy, float sx, float sy, float invw, float invh,
                                         int height, int width, int lineFrom, int lineTo, int maxiter)
        {
            //Console.Out.WriteLine("hello from thread : {0}", blockIdx.x);
            // #pragma omp parallel for
            for(int j = blockIdx.x + lineFrom; j < lineTo; j += gridDim.x)
            {
                // #pragma simd
                for (int i = threadIdx.x; i < width; i += blockDim.x)
                {
                    int tid = j * width + i;
                    float hx = sx  * invw;
                    float hy = sy  * invh;
                    float cx = fx + hx * i;
                    float cy = fy + hy * j;
                    int iter = IterCount(cx, cy, maxiter);
                    output[tid] = iter;
                }
            }
        }

        // on a GPU, it's a better idea to distribute work on a 2D grid
        [EntryPoint]
        public static void Render2D(int[] output,
                                         float fx, float fy, float sx, float sy, float invw, float invh,
                                         int height, int width, int maxiter)
        {
            for (int j = threadIdx.y + blockDim.y * blockIdx.y; j < height; j += gridDim.y * blockDim.y)
            {
                for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < width; i += blockDim.x * gridDim.x)
                {
                    int tid = j * width + i;
                    float hx = sx * invw;
                    float hy = sy * invh;
                    float cx = fx + hx * i;
                    float cy = fy + hy * j;
                    int iter = IterCount(cx, cy, maxiter);
                    output[tid] = iter;
                }
            }
        }
    }
}
