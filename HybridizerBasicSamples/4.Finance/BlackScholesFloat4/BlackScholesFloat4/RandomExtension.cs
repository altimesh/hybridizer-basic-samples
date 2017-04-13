using Hybridizer.Runtime.CUDAImports;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BlackScholesFloat4
{
    static class RandomExtension
    {
        public static float NextFloat(this Random a, float min, float max)
        {
            return ((float)a.NextDouble()) * (max - min) + min;
        }

        public static float4 NextFloat4(this Random a, float min, float max)
        {
            return new float4(a.NextFloat(min, max), a.NextFloat(min, max), a.NextFloat(min, max), a.NextFloat(min, max));
        }
    }

}
