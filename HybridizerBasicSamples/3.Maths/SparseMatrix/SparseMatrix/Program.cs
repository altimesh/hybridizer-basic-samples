using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SparseMatrix
{
    class Program
    {
        static void Main(string[] args)
        {
            string matrixFile, vectorFile;
            ReadArguments(args, out matrixFile, out vectorFile);

            SparseMatrix A = new SparseMatrix(SparseMatrixReader.ReadMatrixFromFile(args[0]));
            float[] X;

            if (!String.IsNullOrEmpty(vectorFile))
            {
                X = VectorReader.ReadVectorFromFile(vectorFile);
            }
            else
            {
                X = VectorReader.GetRandomVector(A.rows.Length - 1);
            }


            float[] B = new float[A.rows.Length - 1];

            Multiply(B, A, X, X.Length);

            for (int i = 0; i < A.rows.Length - 1; ++i)
            {
                Console.WriteLine("{0}", B[i]);
            }
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

        public static void Multiply(float[] res, SparseMatrix m, float[] v, int N)
        {
            for (int i = 0; i < N; ++i)
            {
                uint rowless = m.rows[i];
                uint rowup = m.rows[i + 1];
                float tmp = 0.0F;
                for (uint j = rowless; j < rowup; ++j)
                {
                    tmp += v[m.indices[j]] * m.data[j];
                }

                res[i] = tmp;
            }
        }
    }
}
