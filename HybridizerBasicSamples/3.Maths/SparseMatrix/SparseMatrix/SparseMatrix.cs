using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SparseMatrix
{
    public class SparseMatrix
    {
        public float[] data;
        public uint[] indices;
        public uint[] rows;

        public SparseMatrix (Dictionary<uint, float>[] from)
        {
            rows = new uint[from.Length + 1];
            List<uint> _indices = new List<uint>();
            List<float> _data = new List<float>();

            uint colCounter = 0;
            int i = 0;
            for(; i < from.Length; ++i)
            {
                rows[i] = colCounter;
                foreach(var kvp in from[i])
                {
                    _data.Add(kvp.Value);
                    _indices.Add(kvp.Key);
                    colCounter += 1;
                }
            }
            rows[i] = colCounter;

            indices = _indices.ToArray();
            data = _data.ToArray();
        }
    }
}
