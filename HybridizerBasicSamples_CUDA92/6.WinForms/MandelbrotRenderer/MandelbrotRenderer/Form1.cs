using MandelbrotRenderer.Mandelbrots;
using System;
using System.Drawing;
using System.Windows.Forms;
using System.Drawing.Imaging;
using System.Diagnostics;
using Hybridizer.Runtime.CUDAImports;
using System.Management;
using System.Threading.Tasks;

namespace MandelbrotRenderer
{
    public partial class Form1 : Form
    {
        enum Flavors
        {
            Source,
            AVX,
            AVX2,
            AVX512,
            CUDA
        }

        public byte[,] output;
        Bitmap image { get; set; }
        int maxiter = 50;
        float fromX = -2.0F;
        float fromY = -2.0F;
        float sX = 4.0F;
        float sY = 4.0F;

        HybRunner runnerAVX;
        HybRunner runnerAVX2;
        HybRunner runnerAVX512;
        HybRunner runnerCUDA;
        dynamic MandelbrotAVX;
        dynamic MandelbrotAVX2;
        dynamic MandelbrotAVX512;
        dynamic MandelbrotCUDA;
        
        Flavors flavor = Flavors.Source;

        const int W = 1024;
        const int H = 1024;

        // set the right values depending on your hardware / Hybridizer license
        bool hasCUDA = true;
        bool hasAVX = true;
        bool hasAVX2 = true;
        bool hasAVX512 = true;

        public Form1()
        {
            InitializeComponent();
            CUDA.Enabled = hasCUDA;
            AVX.Enabled = hasAVX;
            AVX2.Enabled = hasAVX2;
            AVX512.Enabled = hasAVX512;

            if (hasCUDA)
            {
                DisplayGPUName();
            }

            ManagementObjectSearcher mos = new ManagementObjectSearcher("root\\CIMV2", "SELECT * FROM Win32_Processor");
            foreach (ManagementObject mo in mos.Get())
            {
                string cpuName = (string)mo["Name"];
                label4.Text = cpuName.Split('@')[0];
            }

            if(hasCUDA)   runnerCUDA = HybRunner.Cuda("MandelbrotRenderer_CUDA.dll").SetDistrib(32, 32, 16, 16, 1, 0);
            if(hasAVX)    runnerAVX = HybRunner.AVX("MandelbrotRenderer_AVX.dll").SetDistrib(Environment.ProcessorCount, 32);
            if(hasAVX2)   runnerAVX2 = HybRunner.AVX("MandelbrotRenderer_AVX2.dll").SetDistrib(Environment.ProcessorCount, 32);
            if(hasAVX512) runnerAVX512 = HybRunner.AVX512("MandelbrotRenderer_AVX512.dll").SetDistrib(Environment.ProcessorCount, 32);

            if(hasCUDA)   MandelbrotCUDA = runnerCUDA.Wrap(new Mandelbrot());
            if(hasAVX)    MandelbrotAVX = runnerAVX.Wrap(new Mandelbrot());
            if(hasAVX2)   MandelbrotAVX2 = runnerAVX2.Wrap(new Mandelbrot());
            if(hasAVX512) MandelbrotAVX512 = runnerAVX512.Wrap(new Mandelbrot());

            this.FormBorderStyle = FormBorderStyle.FixedSingle;
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            image = new Bitmap(W, H, PixelFormat.Format32bppRgb);
            Rendering.Image = image;
            render();
        }

        private void DisplayGPUName()
        {
            int deviceCount;
            cuda.GetDeviceCount(out deviceCount);
            if (deviceCount == 0)
            {
                MessageBox.Show("no CUDA capable device detected -- do not try to use the CUDA version!");
            }

            int major = 0;
            int selectedDevice = 0;
            int mp = 0;
            string deviceName = "";
            for (int i = 0; i < deviceCount; ++i)
            {
                cudaDeviceProp prop;
                cuda.GetDeviceProperties(out prop, i);
                if (prop.major > major)
                {
                    selectedDevice = i;
                    major = prop.major;
                    mp = prop.multiProcessorCount;
                    deviceName = new string(prop.name);
                }
            }

            cuda.SetDevice(selectedDevice);

            label5.Text = deviceName;
        }

        private void render()
        {
            int[] iterCount = new int[W * H];

            Stopwatch watch = new Stopwatch();
            long elapsedMilliseconds;
         
            watch.Start();
            switch (flavor)
            {
                case Flavors.AVX:
                    MandelbrotAVX.Render(iterCount, fromX, fromY, sX, sY, 1.0F / W, 1.0F / H, H, W, 0, H, maxiter);
                    break;
                case Flavors.AVX2:
                    MandelbrotAVX2.Render(iterCount, fromX, fromY, sX, sY, 1.0F / W, 1.0F / H, H, W, 0, H, maxiter);
                    break;
                case Flavors.AVX512:
                    MandelbrotAVX512.Render(iterCount, fromX, fromY, sX, sY, 1.0F / W, 1.0F / H, H, W, 0, H, maxiter);
                    break;
                case Flavors.CUDA:
                    MandelbrotCUDA.Render2D(iterCount, fromX, fromY, sX, sY, 1.0F / W, 1.0F / H, H, W, maxiter);
                    cuda.DeviceSynchronize();
                    break;
                case Flavors.Source:
                default:
                    int slice = H / Environment.ProcessorCount;
                    Parallel.For(0, Environment.ProcessorCount, tid =>
                    {
                        int lineFrom = tid * slice;
                        int lineTo = Math.Min(H, lineFrom + slice);
                        Mandelbrots.Mandelbrot.Render(iterCount, fromX, fromY, sX, sY, 1.0F / W, 1.0F / H, H, W, lineFrom, lineTo, maxiter);
                    });
                    break;
            }
            watch.Stop();

            for (int j = 0; j < H; ++j)
            {
                for (int i = 0; i < W; ++i)
                {
                    int color = GetMandelbrotColor(iterCount[j * W + i], maxiter);
                    image.SetPixel(i, j, Color.FromArgb(color));
                }
            }
            elapsedMilliseconds = watch.ElapsedMilliseconds;
            if(flavor == Flavors.CUDA)
            {
                elapsedMilliseconds = runnerCUDA.LastKernelDuration.ElapsedMilliseconds;
            }

            label2.Text = "Rendering Time : " + watch.ElapsedMilliseconds + " ms";
            
            Rendering.Refresh();
        }
        
        private static int GetMandelbrotColor(int iterCount, int maxiter)
        {
            if (iterCount == maxiter)
            {
                return 0;
            }

            return ((int)(iterCount * (255.0F / (float)(maxiter - 1)))) << 8;
        }

        private void RenderButton_Click(object sender, EventArgs e)
        {
            render();
        }

        private void MaxiterInput_ValueChanged(object sender, EventArgs e)
        {
            if(sender is NumericUpDown)
            {
                NumericUpDown ud = sender as NumericUpDown;
                maxiter = (int) ud.Value;
            }
            render();
        }

        private void FlavorCheckedChanged(object sender, EventArgs e)
        {
            foreach (Control control in this.Flavor.Controls)
            {
                if (control is RadioButton)
                {
                    RadioButton radio = control as RadioButton;
                    if (radio.Checked)
                    {
                        switch (radio.Name.ToLowerInvariant())
                        {
                            case "avx":
                                flavor = Flavors.AVX;
                                break;
                            case "avx2":
                                flavor = Flavors.AVX2;
                                break;
                            case "avx512":
                                flavor = Flavors.AVX512;
                                break;
                            case "cuda":
                                flavor = Flavors.CUDA;
                                break;
                            case "csharp":
                            default:
                                flavor = Flavors.Source;
                                break;
                        }
                        render();
                    }
                }
            }
        }

        private void Rendering_Click(object sender, EventArgs e)
        {
            MouseEventArgs mevent = e as MouseEventArgs;
            if (mevent.Button == MouseButtons.Left) { 
                if(sX/W <= 1.0E-8)
                {
                    MessageBox.Show("float precision limit -- we didn't implement infinite zoom !");
                    return;
                }
            }
            int x = mevent.X;
            int y = mevent.Y;
            float fx = fromX + sX * ((float)x / Rendering.Width);
            float fy = fromY + sY * ((float)y / Rendering.Height);
            if (mevent.Button == MouseButtons.Left)
            {   
                sX *= 0.5F;
                sY *= 0.5F;
            }
            else if(mevent.Button == MouseButtons.Right)
            {
                sX *= 2.0F;
                sY *= 2.0F;
            }

            label3.Text = "Size : " + string.Format("{0:0.##E+00}", sX);
            fromX = fx - 0.5F * sX;
            fromY = fy - 0.5F * sY;
            render();
        }
    }
}
