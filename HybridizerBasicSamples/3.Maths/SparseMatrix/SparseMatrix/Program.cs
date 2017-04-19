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
            VerifyArguments(args);
            SparseMatrix A = new SparseMatrix(ReadMatrixFromFile(args[0]));
            float[] X;
            if (args.Length >= 2)
            {
                X = ReadVectorFromFile(args[1]);
            }
            else
            {
                X = FillRandomVector(A.rows.Length - 1);
            }
            float[] B = new float[A.rows.Length - 1];

            MatrixByVector(B, A, X, X.Length);

            for (int i = 0; i < A.rows.Length - 1; ++i)
            {
                Console.WriteLine("{0}", B[i]);
            }
        }

        private static void VerifyArguments(string[] args)
        {
            if (args.Length < 1)
            {
                throw new ArgumentNullException("no aguments passed ");
            }
            if (!File.Exists(args[0]))
            {
                throw new FileNotFoundException("File doesn't exist");
            }
            if (args.Length >= 2 && !File.Exists(args[1]))
            {
                throw new FileNotFoundException("File doesn't exist");
            }
        }

        public static void MatrixByVector(float[] res, SparseMatrix m, float[] v, int N)
        {
            for(int i = 0; i < N; ++i)
            {
                uint rowless = m.rows[i];
                uint rowup = m.rows[i+1];
                float tmp = 0.0F;
                for (uint j = rowless; j < rowup; ++j)
                {
                    tmp += v[m.indices[j]] * m.data[j];
                }

                res[i] = tmp;
            }
        }
        
        enum MatrixType
        {
            General, 
            Symmetric, 
            SkewSymmetric,
            Hermitian
        };

        public static Dictionary<uint,float>[] ReadMatrixFromFile(string filePath)
        {
            string path = GetMatrixPath(filePath);

            int rowCount, colCount;
            string line;
            Dictionary<uint, Dictionary<uint, float>> tmp = new Dictionary<uint, Dictionary<uint, float>>();

            StreamReader reader = new StreamReader(path);

            line = reader.ReadLine().ToLowerInvariant();
            MatrixType matrixType = ReadMatrixType(line);
            while ((line = reader.ReadLine()) != null && line.StartsWith("%")) { }
            
            ReadMatrixSize(line, out rowCount, out colCount);

            while ((line = reader.ReadLine()) != null)
            {
                string[] lineSplitted = line.Split(' ');
                uint row, col; float data;
                ReadData(lineSplitted, out row, out col, out data);
                InsertValue(tmp, row, col, data);

                switch (matrixType)
                {
                    case MatrixType.General:
                        break;
                    case MatrixType.Symmetric:
                        InsertValue(tmp, col, row, data);
                        break;
                    case MatrixType.SkewSymmetric:
                        InsertValue(tmp, col, row, -data);
                        break;
                    default:
                        throw new NotImplementedException("only symmetric, general, and skew-symmetric matrices are supported");
                }
            }

            return BuildRawSparseMatrix(rowCount, tmp);
        }

        private static string GetMatrixPath(string filePath)
        {
            string path;
            if (Path.IsPathRooted(filePath))
            {
                path = filePath;
            }
            else
            {
                path = Path.Combine(Environment.CurrentDirectory, filePath);
            }

            if (!File.Exists(path))
            {
                throw new FileNotFoundException(path);
            }

            return path;
        }

        private static MatrixType ReadMatrixType(string line)
        {
            if (line.Contains("general"))
            {
                return MatrixType.General;
            }
            else if (line.Contains("symmetric"))
            {
                return MatrixType.Symmetric;
            }
            else if (line.Contains("skew-symmetric"))
            {
                return MatrixType.SkewSymmetric;
            }
            else if (line.Contains("hermitian"))
            {
                return MatrixType.Hermitian;
            }
            else
            {
                throw new ApplicationException("invalid input file -- matrix type not recognized");
            }
        }

        private static void ReadMatrixSize(string line, out int rowCount, out int colCount)
        {
            string[] lineSplitted = line.Split(' ');
            if (lineSplitted.Length != 3)
            {
                throw new ApplicationException("cannot read matrix size");
            }
            else
            {
                if (!int.TryParse(lineSplitted[0], out rowCount))
                {
                    throw new ApplicationException("cannot read matrix size");
                }
                if (!int.TryParse(lineSplitted[1], out colCount))
                {
                    throw new ApplicationException("cannot read matrix size");
                }
            }
        }

        private static void ReadData(string[] lineSplitted, out uint row, out uint col, out float data)
        {
            if (!UInt32.TryParse(lineSplitted[0], out row))
            {
                throw new ApplicationException("invalid line");
            }
            if (!UInt32.TryParse(lineSplitted[1], out col))
            {
                throw new ApplicationException("invalid line");
            }
            if (!Single.TryParse(lineSplitted[2],NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out data))
            {
                throw new ApplicationException("invalid line");
            }
        }

        private static void InsertValue(Dictionary<uint, Dictionary<uint, float>> res, uint row, uint col, float data)
        {
            if (!res.ContainsKey(row))
            {
                res.Add(row, new Dictionary<uint, float>());
                res[row].Add(col, data);
            }
        }

        private static Dictionary<uint, float>[] BuildRawSparseMatrix(int rowCount, Dictionary<uint, Dictionary<uint, float>> tmp)
        {
            Dictionary<uint, float>[] res = new Dictionary<uint, float>[rowCount];

            for (uint i = 0; i < rowCount; ++i)
            {
                if (tmp.ContainsKey(i))
                {
                    res[i] = tmp[i];
                }
                else
                {
                    res[i] = new Dictionary<uint, float>();
                }
            }

            return res;
        }

        public static float[] FillRandomVector(int size)
        {
            float[] res = new float[size];
            Random rand = new Random(Guid.NewGuid().GetHashCode());
            for (int i = 0; i < size; ++i)
            {
                res[i] = rand.NextFloat();
            }
            return res;
        }

        public static float[] ReadVectorFromFile(String filePath)
        {
            string path = GetMatrixPath(filePath);

            int rowCount;
            string line;

            StreamReader reader = new StreamReader(path);

            while ((line = reader.ReadLine()) != null && line.StartsWith("%")) { }

            ReadVectorSize(line, out rowCount);

            float[] res = new float[rowCount];
            int cpt = 0;
            while ((line = reader.ReadLine()) != null)
            {
                string[] lineSplitted = line.Split(' ');
                float data;
                ReadData(lineSplitted, out data);
                res[cpt] = data;
                ++cpt;
            }

            return res;
        }

        private static void ReadVectorSize(string line, out int rowCount)
        {
            string[] lineSplitted = line.Split(' ');
            Console.WriteLine(lineSplitted.Length);
            if (lineSplitted.Length != 2)
            {
                throw new ApplicationException("cannot read matrix size");
            }
            else
            {
                if (!int.TryParse(lineSplitted[0], out rowCount))
                {
                    throw new ApplicationException("cannot read matrix size");
                }
            }
        }

        private static void ReadData(string[] lineSplitted, out float data)
        {
            if (!Single.TryParse(lineSplitted[0], NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out data))
            {
                throw new ApplicationException("invalid line");
            }
        }
    }
}
