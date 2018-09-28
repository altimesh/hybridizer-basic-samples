using Hybridizer.Runtime.CUDAImports;
using System.Drawing;
using System.Diagnostics;
using System;
using System.Threading.Tasks;

namespace Mandelbrot
{
    class Program
    {
        const int N = 2048;
        
        static double input_size = 4.0;
        static int input_maxiter = 32;
        static double input_fromX = -2.0;
        static double input_fromY = -2.0;

        [Kernel]
        public static int IterCount(double cx, double cy, int maxiter)
        {
            int result = 0;
            double x = 0.0;
            double y = 0.0;
            double xx = 0.0, yy = 0.0;
            while (xx + yy <= 4.0 && result < maxiter)
            {
                xx = x * x;
                yy = y * y;
                double xtmp = xx - yy + cx;
                y = 2.0 * x * y + cy;
                x = xtmp;
                result++;
            }

            return result;
        }

        [EntryPoint("run")]
        public static void Run(int[] light, int lineFrom, int lineTo, double fromX, double fromY, double h, int maxiter)
        {
            for (int line = lineFrom + blockIdx.x ; line < lineTo; line += gridDim.x)
            {
                for (int j = threadIdx.x ; j < N; j += blockDim.x)
                {
                    double x = fromX + line * h;
                    double y = fromY + j * h;
                    light[line * N + j] = IterCount(x, y, maxiter);
                }
            }
        }

        private static dynamic wrapper;

        public static void ComputeImage(int[] light, bool accelerate = true)
        {
            double input_h = input_size/(double)N;
            if (accelerate)
            {
                wrapper.Run(light, 0, N, input_fromX, input_fromY, input_h, input_maxiter);
            }
            else
            {
                // This is the Plain C# version, using parallel-for to parallelize work
                Parallel.For(0, N, (line) =>
                {
                    Run(light, line, line + 1, input_fromX, input_fromY, input_h, input_maxiter);
                });
            }
        }
        
        static void Usage()
        {
            Console.Out.WriteLine("Usage : manelbrot-double.exe [fromX] [fromY] [size] [flavor]");
        }
        
        // fromX, fromY, h, size, flavor : C# or AVX512
        static void Main(string[] args)
        {
            // null flavor means all
            string flavor = null;
            string filename = "mandelbrot.png";
            
            try   
            {
                if (args.Length > 0)
                    input_fromX = double.Parse(args[0]);
                if (args.Length > 1)
                    input_fromY = double.Parse(args[1]);
                if (args.Length > 2)
                    input_size = double.Parse(args[2]);
                if (args.Length > 3)
                    input_maxiter = int.Parse(args[3]);
                if (args.Length > 4)
                    flavor = args[4];
                if (args.Length > 5)
                    filename = args[5];
            }
            catch (Exception ex)
            {
                Usage();
                Environment.Exit(1);
            }
                        
            Stopwatch cstimer = new Stopwatch();
            Stopwatch avx512timer = new Stopwatch();
            
            const int redo = 10;

            int[] light_net = new int[N * N];
            int[] light_avx512 = new int[N * N];
            
            if ((flavor == null) || ("C#".Equals(flavor)))
            {
                Console.Out.WriteLine("Running plain C#");
                
                cstimer.Start();
    
                // cold run
                ComputeImage(light_net, false);
            
                #region C#
                for (int i = 0; i < redo; ++i)
                {
                    ComputeImage(light_net, false);
                }
                #endregion C#
                
                cstimer.Stop();
            
                Console.Out.WriteLine("Plain C# Took {0} ms", cstimer.ElapsedMilliseconds) ;
            }
            
            if ((flavor == null) || ("AVX512".Equals(flavor)))
            {
                Console.Out.WriteLine("Building Wrapper");

                HybRunner runner = HybRunner.AVX512().SetDistrib(Environment.ProcessorCount,32);
                wrapper = runner.Wrap(new Program());
            
                Console.Out.WriteLine("Running with AVX512");
            
                // cold run
                ComputeImage(light_avx512, true);
            
                avx512timer.Start();
                
                #region avx512
                for (int i = 0; i < redo; ++i)
                {
                    ComputeImage(light_avx512, true);
                }
                #endregion
                
                avx512timer.Stop();
                
                Console.Out.WriteLine("AVX512 Took {0} ms", avx512timer.ElapsedMilliseconds) ;
            }

            Console.Out.WriteLine("Saving to file");

            #region save to image
            Color[] colors = new Color[input_maxiter + 1];

            for (int k = 0; k < input_maxiter; ++k)
            {
                int red = (int) (127.0 * (double)k / (double)input_maxiter);
                int green = (int)(200.0 * (double)k / (double)input_maxiter);
                int blue = (int)(90.0 * (double)k / (double)input_maxiter);
                colors[k] = Color.FromArgb(red, green, blue);
            }
            colors[input_maxiter] = Color.Black;

            if ("C#".Equals(flavor))
            {
                light_avx512 = light_net ;
            }
            
            Bitmap image = new Bitmap(N, N);
            for (int i = 0; i < N; ++i)
            {
                for (int j = 0; j < N; ++j)
                {
                    int index = i * N + j;
                    image.SetPixel(i, j, colors[light_avx512[index]]);
                }
            }

            image.Save(filename, System.Drawing.Imaging.ImageFormat.Png);
            #endregion
                
            Console.Out.WriteLine("DONE");
        }
    }
}
