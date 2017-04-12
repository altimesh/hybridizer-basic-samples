using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharedMatrix
{
    public class NaiveMatrix
    {
        private float[] values;

        public int Height { get; private set; }
        public int Width { get; private set; }
        private static Random rand = new Random();

        public NaiveMatrix(int width = 1024, int height = 1024)
        {
            this.Height = height;
            this.Width = width;
            this.values = new float[height * width];
        }

        public float this[int i]
        {
            get { return this.values[i]; }

            set { this.values[i] = value; }
        }

        public void FillMatrix()
        {
            for (int i = 0; i < this.Height; ++i)
            {
                for (int j = 0; j < this.Width; ++j)
                {
                    this[i * this.Width + j] = (float)rand.NextDouble();
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

        public bool IsSame(NaiveMatrix m)
        {
            if (m == this)
                return true;

            if (m.GetType() != typeof(NaiveMatrix))
                return false;
            
            if (this.Height != m.Height || this.Width != m.Width)
                return false;

            for (int i = 0; i < this.Height; ++i)
            {
                for (int j = 0; j < this.Width; ++j)
                {
                    if (this[i * this.Width + j] != m[i * m.Width + j])
                    {
                        Console.WriteLine("Error at (" + i + "," + j + "): expected " + this[i * this.Width + j] + " and got " + m[i * m.Width + j]);
                        return false;
                    }
                }
            }

            return true;
        }
    }
}
