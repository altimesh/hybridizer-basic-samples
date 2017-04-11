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

        public class NaiveMatrix
        {
            private int height = 1024;
            private int width = 1024;
            private float[] values;

            public NaiveMatrix(int width, int height)
            {
                this.height = height;
                this.width = width;
                this.values = new float[this.height * this.width];
            }

            public NaiveMatrix()
            {
                this.values = new float[this.height * this.width];
            }

            public int Height
            {
                get { return this.height; }
                set { this.height = value; }
            }

            public int Width
            {
                get { return this.width; }
                set { this.width = value; }
            }

            public float this[int i]
            {
                get { return this.values[i]; }

                set { this.values[i] = value; }
            }


            public void FillMatrix()
            {
                Random rand = new Random();
                for (int i = 0; i < this.height; ++i)
                {
                    for (int j = 0; j < this.width; ++j)
                    {
                        this[i * this.width + j] = (float)rand.Next(2);
                    }

                }
            }

            public void WriteMatrix()
            {
                for (int k = 0; k < this.Height; ++k)
                {
                    for (int j = 0; j < this.Width; ++j)
                    {
                        Console.Write(this[k * this.Width + j].ToString() + " ");
                    }
                    Console.WriteLine("");
                }
            }

            override
            public Boolean Equals(Object o)
            {
                if (o == this) return true;
                if (o.GetType() == typeof(NaiveMatrix))
                {
                    NaiveMatrix m = (NaiveMatrix)o;
                    Boolean b = this.height == m.height && this.height == m.height;
                    if (!b)
                    {
                        for (int i = 0; i < this.height; ++i)
                        {
                            for (int j = 0; j < this.width; ++j)
                            {
                                b = b && this[i * this.width + j] == m[i * m.width + j];
                            }
                        }
                    }
                    return b;
                }

                return false;
            }
        }


        static void Main(string[] args)
        {
            if (args.Length == 0)
            {
                args = new string[] { "1024", "1024", "1024", "1024" };
            }
            const int redo = 10;
            int heightA = Convert.ToInt32(args[0]);
            int widthA = Convert.ToInt32(args[1]);
            int heightB = Convert.ToInt32(args[2]);
            int widthB = Convert.ToInt32(args[3]);
            if (widthA != heightB)
            {
                throw new ArgumentException();
            }

            NaiveMatrix matrixA = new NaiveMatrix(widthA, heightA);
            NaiveMatrix matrixB = new NaiveMatrix(widthB, heightB);

            NaiveMatrix res_net = new NaiveMatrix(widthB, heightA);
            NaiveMatrix res_cuda = new NaiveMatrix(widthB, heightA);

            double numberCompute = (double)matrixA.Height * (double)matrixA.Width * (double)matrixB.Width * 3.0;

            Stopwatch watch = new Stopwatch();

            matrixA.FillMatrix();
            matrixB.FillMatrix();

            Random rand = new Random();

            #region CUDA 

            HybRunner runner = HybRunner.Cuda("NaiveMatrix_CUDA.dll").SetDistrib(4, 5, 8, 128, 1, 0);
            dynamic wrapper = runner.Wrap(new Program());

            watch.Start();

            for (int i = 0; i < redo; ++i)
            {
                wrapper.ComputeRowsOfProduct(res_cuda, matrixA, matrixB, 0, matrixA.Height);
            }

            watch.Stop();

            Console.WriteLine("CUDA float operation/s : {0}", numberCompute * (double)redo / (watch.ElapsedMilliseconds * 1.0E-03));
            Console.WriteLine("without memcpy         : {0}", numberCompute / (runner.LastKernelDuration.ElapsedMilliseconds * 1.0E-03));
            #endregion

            #region C#
            watch.Restart();

            for (int i = 0; i < redo; ++i)
            {
                Parallel.For(0, matrixA.Height, (line) =>
            {
                ComputeRowsOfProduct(res_net, matrixA, matrixB, line, line + 1);
            });
            }
            watch.Stop();
            Console.WriteLine("C#   float operation/s :   {0}", numberCompute * (double)redo / (watch.ElapsedMilliseconds * 1.0E-03));
            #endregion

            // verify the results

            if (!res_net.Equals(res_cuda))
                Console.Out.WriteLine("ERROR !");


            //Write the matrix on the console
            res_cuda.WriteMatrix();

            Console.Out.WriteLine("DONE");


        }

        [EntryPoint]
        public static void ComputeRowsOfProduct(NaiveMatrix resultMatrix, NaiveMatrix matrixA, NaiveMatrix matrixB, int lineFrom, int lineTo)
        {
            int commonSize = matrixA.Width;
            for (int i = lineFrom + threadIdx.y + blockIdx.y * blockDim.y; i < lineTo; i += blockDim.y * gridDim.y)
            {
                for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < matrixB.Width; j += blockDim.x * gridDim.x)
                {
                    resultMatrix[i * matrixB.Width + j] = 0.0f;

                    for (int k = 0; k < commonSize; ++k)
                    {
                        resultMatrix[i * matrixB.Width + j] += (matrixA[i * commonSize + k] * matrixB[k * matrixB.Width + j]);

                    }
                }
            }
        }
    }
}
