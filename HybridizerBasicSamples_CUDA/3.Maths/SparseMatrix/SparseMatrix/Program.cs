﻿using Hybridizer.Runtime.CUDAImports;
using System;
using Hybridizer.Basic.Utilities;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Hybridizer.Basic.Maths
{
    class Program
    {
        static void Main(string[] args)
        {

            SparseMatrix A = SparseMatrix.Laplacian_1D(10000000);

            float[] X = VectorReader.GetSplatVector(10000000, 1.0F);

            int redo = 2;
            double memoryOperationsSize = (double) redo * (3.0 * (double) (A.data.Length * sizeof(float)) + (double) (2 * A.rows.Length * sizeof(uint)) + (double) (A.indices.Length * sizeof(uint)));
            Console.WriteLine("matrix read --- starting computations");

            float[] B = new float[A.rows.Length - 1];

            #region CUDA
            cudaDeviceProp prop;
            cuda.GetDeviceProperties(out prop, 0);

            HybRunner runner = HybRunner.Cuda().SetDistrib(8 * prop.multiProcessorCount, 256);
            dynamic wrapper = runner.Wrap(new Program());

            for (int i = 0; i < redo; ++i)
            {
                wrapper.Multiply(B, A, X, X.Length);
            }
            #endregion
        }

        private static void ReadArguments(string[] args, out string matrixFile, out string vectorFile)
        {
            if (args.Length < 1)
            {
                throw new ArgumentNullException("no arguments passed ");
            }
            if (!File.Exists(args[0]))
            {
                throw new FileNotFoundException("File doesn't exist");
            }
            if (args.Length >= 2 && File.Exists(args[1]))
            {
                vectorFile = args[1];
            }
            else
            {
                vectorFile = null;
            }
            matrixFile = args[0];

        }

        [EntryPoint]
        public static void Multiply(float[] res, SparseMatrix m, float[] v, int N)
        {
            Parallel.For(0, N, (i) =>
            {
                uint rowless = m.rows[i];
                uint rowup = m.rows[i + 1];
                float tmp = 0.0F;
                for (uint j = rowless; j < rowup; ++j)
                {
                    tmp += v[m.indices[j]] * m.data[j];
                }
                res[i] = tmp;
            });
        }
    }
}
