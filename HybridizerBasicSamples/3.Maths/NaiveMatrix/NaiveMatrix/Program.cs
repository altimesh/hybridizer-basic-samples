using Hybridizer.Runtime.CUDAImports;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NaiveMatrix
{
    class Program
    {
        //first matrix height number
        const int height1 = 1024;
        //second matrix width number
        const int width2 = 1024;
        //the common size between the two matrix to allow the multiplication (first matrix width = second matrix height)
        const int commonSize = 128;

        static void Main(string[] args)
        {
            const int redo = 10;

            float[] matrix1 = new float[commonSize * height1];
            float[] matrix2 = new float[width2 * commonSize];

            float[] res_net = new float[width2 * height1];
            float[] res_cuda = new float[width2 * height1];

            Stopwatch watch = new Stopwatch();

            FillMatrix(matrix1, matrix2);

            #region CUDA 

            HybRunner runner = HybRunner.Cuda("NaiveMatrix_CUDA.dll").SetDistrib(4, 5, 8, 128, 1, 0);
            dynamic wrapper = runner.Wrap(new Program());

            for (int i = 0; i < redo; ++i)
            {
                if (i == 1) watch.Start();
                wrapper.MultiplyMatrix(res_cuda, matrix1, matrix2, 0, height1);
            }

            watch.Stop();

            Console.WriteLine("CUDA float operation/s : {0}", (double)width2 * (double)height1 * ((double)redo - 1.0) / (watch.ElapsedMilliseconds * 1.0E-03));
            Console.WriteLine("without memcpy         : {0}", (double)width2 * (double)height1 / (runner.LastKernelDuration.ElapsedMilliseconds * 1.0E-03));
            #endregion

            #region C#
            watch.Restart();

            for (int i = 0; i < redo; ++i)
            {
                Parallel.For(0, height1, (line) =>
            {
                MultiplyMatrix(res_net, matrix1, matrix2, line, line + 1);
            });
            }
            watch.Stop();
            Console.WriteLine("C#   float operation/s : {0}", (double)width2 * (double)height1 * (double)redo / (watch.ElapsedMilliseconds * 1.0E-03));
            #endregion

            // verify the results
            for (int k = 0; k < height1; ++k)
            {
                for (int j = 0; j < width2; ++j)
                {
                    if (res_net[k*width2 + j] != res_cuda[k * width2 + j])
                        Console.Out.WriteLine("ERROR !");
                }
            }
            Console.Out.WriteLine("DONE");


        }

        public static void FillMatrix(float[] matrix1, float[] matrix2)
        {
            Random rand = new Random();
            for (int i = 0; i < commonSize; ++i)
            {
                for (int j = 0; j < height1; ++j)
                {
                    matrix1[j * commonSize + i] = (float)rand.Next(2);
                }

                for (int j = 0; j < width2; ++j)
                {
                    matrix2[i * width2 + j] = (float)rand.Next(2);
                }
            }
        }

        [EntryPoint]
        public static void MultiplyMatrix(float[] resultMatrix, float[] matrix1, float[] matrix2, int lineFrom, int lineTo)
        {
            for (int i = lineFrom + threadIdx.y + blockIdx.y * blockDim.y; i < lineTo; i += blockDim.y * gridDim.y)
            {
                for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < width2; j += blockDim.x * gridDim.x)
                { 
                    resultMatrix[j + i * width2] = 0;

                    for (int k = 0; k < commonSize; ++k)
                    {
                        resultMatrix[j + i * width2] += matrix1[i * commonSize + k] * matrix2[k * width2 + j];
                    }
                }
            }
        }
    }
}
