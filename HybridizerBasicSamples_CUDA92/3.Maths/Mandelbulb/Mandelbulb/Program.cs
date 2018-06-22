using Hybridizer.Runtime.CUDAImports;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Mandelbulb
{
    class Program
    {
        static void Main(string[] args)
        {
            Rendering.BitMap("mandelbulb.png", 1024, 1024);
        }
    }
}
