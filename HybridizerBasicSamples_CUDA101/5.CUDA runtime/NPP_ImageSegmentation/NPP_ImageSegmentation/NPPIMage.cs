using Hybridizer.Runtime.CUDAImports;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace NPP_ImageSegmentation
{
    class NPPImage : IDisposable
    {
        public ushort[] hostData { get; set; }
        public IntPtr deviceData { get; set; }
        public int width { get; set; }
        public int height { get; set; }
        // TODO: line alignment. 
        public int pitch { get; set; }
        private bool disposed = false;

        public static NPPImage Load(string path, cudaStream_t stream)
        {
            NPPImage result = new NPPImage();
            byte[] rawData;
            if (Path.GetExtension(path).Contains("pgm"))
            {
                using (FileStream fs = new FileStream(path, FileMode.Open))
                {
                    using (TextReader tReader = new StreamReader(fs))
                    using (BinaryReader bReader = new BinaryReader(fs))
                    {
                        string formatLine = tReader.ReadLine(); // skip
                        string sizeLine = tReader.ReadLine();
                        string[] splitted = sizeLine.Split(' ');
                        result.width = int.Parse(splitted[0]);
                        result.height = int.Parse(splitted[1]);

                        string maxValueLine = tReader.ReadLine(); // skip
                        int pos = formatLine.Length + sizeLine.Length + maxValueLine.Length + 3;
                        fs.Seek(pos, SeekOrigin.Begin);

                        // TODO: optimize that part
                        rawData = bReader.ReadBytes((int)(fs.Length - pos));

                    }
                }
            }
            else if (Path.GetExtension(path).Contains("png"))
            {
                Bitmap image = Bitmap.FromFile(path) as Bitmap;
                result.width = image.Width;
                result.height = image.Height;
                rawData = new byte[result.width * result.height];
                int index = 0;
                for (int j = 0; j < result.height; ++j)
                {
                    for (int i = 0; i < result.width; ++i, ++index)
                    {
                        rawData[index] = image.GetPixel(i, j).R;
                    }
                }
            }
            else
            {
                throw new NotSupportedException("unsupported file format");
            }

            IntPtr deviceData;
            size_t p;
            cuda.ERROR_CHECK(cuda.MallocPitch(out deviceData, out p, result.width * sizeof(ushort), result.height));
            result.pitch = (int)p;

            result.hostData = new ushort[result.height * result.width];
            for (int j = 0; j < result.height; ++j)
            {
                for (int i = 0; i < result.width; ++i)
                {
                    result.hostData[j * result.width + i] = rawData[j * result.width + i];
                }
            }

            var handle = GCHandle.Alloc(result.hostData, GCHandleType.Pinned);
            cuda.ERROR_CHECK(cuda.Memcpy2DAsync(deviceData, p, handle.AddrOfPinnedObject(), result.width * sizeof(ushort), result.width * sizeof(ushort), result.height, cudaMemcpyKind.cudaMemcpyHostToDevice, stream));
            handle.Free();
            result.deviceData = deviceData;

            return result;
        }

        public void Save(string path)
        {
            Bitmap bitmap = new Bitmap(width, height);
            int index = 0;
            for (int j = 0; j < height; ++j)
            {
                for (int i = 0; i < width; ++i, ++index)
                {
                    ushort pixel = hostData[j * pitch + i];
                    bitmap.SetPixel(i, j, Color.FromArgb(pixel, pixel, pixel));
                }
            }
            bitmap.Save(path, ImageFormat.Png);
        }

        public static void Save(string path, IntPtr devicePtr, int width, int pitch, int height)
        {
            ushort[] arr = new ushort[height * width];
            var handle = GCHandle.Alloc(arr, GCHandleType.Pinned);
            cuda.ERROR_CHECK(cuda.Memcpy2D(handle.AddrOfPinnedObject(), width * sizeof(ushort), devicePtr, pitch, width * sizeof(ushort), height, cudaMemcpyKind.cudaMemcpyDeviceToHost));
            handle.Free();
            Bitmap bitmap = new Bitmap(width, height);
            int index = 0;
            for (int j = 0; j < height; ++j)
            {
                for (int i = 0; i < width; ++i, ++index)
                {
                    ushort pixel = arr[j * width + i];
                    bitmap.SetPixel(i, j, Color.FromArgb(pixel, pixel, pixel));
                }
            }
            bitmap.Save(path, ImageFormat.Png);
        }

        public static void Save(string path, uchar4[] colors, int width, int height)
        {
            Bitmap bitmap = new Bitmap(width, height);
            int index = 0;
            for (int j = 0; j < height; ++j)
            {
                for (int i = 0; i < width; ++i, ++index)
                {
                    uchar4 pixel = colors[j * width + i];
                    bitmap.SetPixel(i, j, Color.FromArgb(pixel.x, pixel.y, pixel.z));
                }
            }

            bitmap.Save(path, ImageFormat.Png);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposed)
            {
                if (disposing)
                {
                    if (deviceData != IntPtr.Zero)
                    {
                        cuda.Free(deviceData);
                        deviceData = IntPtr.Zero;
                    }
                }

                disposed = true;
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
    }
}
