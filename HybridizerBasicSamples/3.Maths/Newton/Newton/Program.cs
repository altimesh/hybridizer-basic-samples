using Hybridizer.Runtime.CUDAImports;
using System.Drawing;
using System.Diagnostics;
using System;
using System.Threading.Tasks;

namespace Mandelbrot
{
    class Program
    {
        const int maxiter = 1024;
        const int N = 2048;
        const double fromX = -1.0;
        const double fromY = -1.0;
        const double size = 2.0;
        const double h = size / (double)N;
        const double tol = 0.0000001;

        [Kernel]
        public static int IterCount(double cx, double cy, out int itercount)
        {
            itercount = 0;
            double x = cx;
            double y = cy;
            double xx = 0.0, xy = 0.0, yy = 0.0, xxy = 0.0, xyy = 0.0, xxx = 0.0, yyy = 0.0, yyyy = 0.0, xxxx = 0.0, xxxxx = 0.0;
            while (itercount < maxiter)
            {
                xy = x * y;
                xx = x * x;
                yy = y * y;
                xyy = x * yy;
                xxy = xx * y;
                xxx = xx * x;
                yyy = yy * y;
                xxxx = xx * xx;
                yyyy = yy * yy;
                xxxxx = xxx * xx;

                double invdenum = 1.0 / (3.0 * xxxx + 6.0 * xx * yy + 3.0 * yyyy);

                double numreal = 2.0 * xxxxx + 4.0 * xxx * yy + xx + 2.0 * x * yyyy - yy;
                double numim = 2.0 * xxxx * y + 4.0 * xx * yyy - 2.0 * x * y + 2.0 * yyy * yy;

                x = numreal * invdenum;
                y = numim * invdenum;
                itercount++;

                int root = RootFind(x, y);
                if (root > 0)
                {
                    return root;
                }
            }

            return 0;
        }

        [Kernel]
        public static int RootFind(double x, double y)
        {
            double sqrtRoot = Math.Sqrt(3.0 / 4.0);

            if ((x <= 1.0 + tol && x >= 1.0 - tol && y <= 0.0 + tol && y >= 0.0 - tol))
            {
                return 1;
            }
            else if ((x <= -0.5 + tol && x >= -0.5 - tol && y <= sqrtRoot + tol && y >= sqrtRoot - tol))
            {
                return 2;
            }
            else if ((x <= -0.5 + tol && x >= -0.5 - tol && y <= -sqrtRoot + tol && y >= -sqrtRoot - tol))
            {
                return 3;
            }

            return 0;
        }

        [EntryPoint("run")]
        public static void Run(int[] rootindex, int lineFrom, int lineTo, int[] itercount)
        {
            for (int i = lineFrom + threadIdx.y + blockIdx.y * blockDim.y; i < lineTo; i += blockDim.y * gridDim.y)
            {
                for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < N; j += blockDim.x * gridDim.x)
                 {
                    double x = fromX + i * h;
                    double y = fromY + j * h;
                    rootindex[i * N + j] = IterCount(x, y, out itercount[i * N + j]);
                }
            }
        }

        private static dynamic wrapper;

        public static void ComputeImage(int[] light,int[] rootColor, bool accelerate = true)
        {
            if (accelerate)
            {
                wrapper.Run(light, 0, N, rootColor);
            }
            else
            {
                Parallel.For(0, N, (line) =>
                {
                    Run(light, line, line + 1, rootColor);
                });
            }
        }

        static int ComputeLight(int iter)
        {
            return System.Math.Min(iter*16,255);
        }

        static void Main(string[] args)
        {
            const int redo = 10;
            int[] itercount_net = new int[N * N];
            int[] rootindex_net = new int[N * N];
            int[] itercount_cuda = new int[N * N];
            int[] rootindex_cuda = new int[N * N];

            #region c#

            Stopwatch watch = new Stopwatch();
            watch.Start();

            for (int i = 0; i < redo; ++i)
            {
                ComputeImage(rootindex_net, itercount_net, false);
            }

            watch.Stop();
            double firstWatchResult = 1.0E-6 * ((double)(N * N) * (double)redo / (1.0E-3 * watch.ElapsedMilliseconds));

            #endregion c#

            HybRunner runner = HybRunner.Cuda("Newton_CUDA.dll").SetDistrib(4, 5, 8, 128, 1, 0);
            wrapper = runner.Wrap(new Program());

            #region cuda
            
            for (int i = 0; i < redo; ++i)
            {
                if(i == 1) // skip first call to skip cubin link phase
                    watch.Restart();

                ComputeImage(rootindex_cuda, itercount_cuda, true);
            }
            watch.Stop();

            Console.WriteLine("C#   MPixels/s :  {0}", firstWatchResult);
            Console.WriteLine("CUDA MPixels/s :  {0}", 1.0E-6 * ((double)(N * N) * (double)redo / (1.0E-3 * watch.ElapsedMilliseconds)));
            Console.WriteLine("without memcpy :  {0}", 1.0E-6 * ((double)(N * N) / (1.0E-3 * runner.LastKernelDuration.ElapsedMilliseconds)));

            #endregion

            #region save to image

            Bitmap image = new Bitmap(N, N);

            for (int i = 0; i < N; ++i)
            {
                for (int j = 0; j < N; ++j)
                {

                    int index = i * N + j;
                    int root = rootindex_net[index];
                    int light = ComputeLight(itercount_net[index]);

                    switch (root)
                    {
                        case 0:
                            image.SetPixel(i, j, Color.Black);
                            break;
                        case 1:
                            image.SetPixel(i, j, Color.FromArgb(light, 0 ,0));
                            break;
                        case 2:
                            image.SetPixel(i, j, Color.FromArgb(0, 0, light));
                            break;
                        case 3:
                            image.SetPixel(i, j, Color.FromArgb(0, light, 0));
                            break;
                        default:
                            throw new ApplicationException();
                    }
                }
            }

            image.Save("newton.png", System.Drawing.Imaging.ImageFormat.Png);
            #endregion

            Process.Start("newton.png");
        }
    }
}

