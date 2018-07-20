using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;

namespace TextureAndSurface
{
    [StructLayout(LayoutKind.Explicit, Size=14, Pack=2)]
    unsafe struct FileHeader
    {
        [FieldOffset(0)]
        public fixed byte magic[2]; // = 0x42 0x4D
        [FieldOffset(2)]
        public uint fileSizeInBytes;
        [FieldOffset(6)]
        public ushort reserved0;
        [FieldOffset(8)]
        public ushort reserved1;
        [FieldOffset(10)]
        public uint pixelArrayOffset;

        public static FileHeader GrayBitmap(uint width, uint height)
        {
            FileHeader res = new FileHeader();
            res.magic[0] = 0x42;
            res.magic[1] = 0x4D;
            res.fileSizeInBytes = width * height + 54 + 1024;
            res.pixelArrayOffset = 54 + 1024;
            return res;
        }
    }

    enum BitmapCompression : uint
    {
        BI_RGB = 0, // none -- Most common
        BI_RLE8 = 1, // RLE 8-bit/pixel -- Can be used only with 8-bit/pixel bitmaps
        BI_RLE4 = 2, // RLE 4-bit/pixel -- Can be used only with 4-bit/pixel bitmaps
        BI_BITFIELDS = 3, // OS22XBITMAPHEADER: Huffman 1D BITMAPV2INFOHEADER: RGB bit field masks, BITMAPV3INFOHEADER+: RGBA
        BI_JPEG = 4, // OS22XBITMAPHEADER: RLE-24 BITMAPV4INFOHEADER+: JPEG image for printing[12] 
        BI_PNG = 5, // BITMAPV4INFOHEADER+: PNG image for printing[12]
        BI_ALPHABITFIELDS = 6, // RGBA bit field masks -- only Windows CE 5.0 with .NET 4.0 or later
        BI_CMYK = 11, //none -- only Windows Metafile CMYK[3]
        BI_CMYKRLE8 = 12, // RLE-8 -- only Windows Metafile CMYK
        BI_CMYKRLE4 = 13, // RLE-4 -- only Windows Metafile CMYK
    }

    [StructLayout(LayoutKind.Explicit, Size = 40, Pack = 2)]
    unsafe struct BitmapInfoHeader
    {
        [FieldOffset(0)]
        public uint cwSize; // = 40
        [FieldOffset(4)]
        public uint width;
        [FieldOffset(8)]
        public uint height;
        [FieldOffset(12)]
        public ushort colorPlanes; // = 1
        [FieldOffset(14)]
        public ushort bitsPerPixel;
        [FieldOffset(16)]
        public BitmapCompression compression;
        [FieldOffset(20)]
        public uint imageSize; // size of raw bitmap data
        [FieldOffset(24)]
        public uint hrez; // pixels per meter
        [FieldOffset(28)]
        public uint vrez; // pixels per meter
        [FieldOffset(32)]
        public uint numberOfColorsInPalette;
        [FieldOffset(36)]
        public uint numberOfImportantColors;

        public static BitmapInfoHeader GrayBitmap(uint width, uint height)
        {
            BitmapInfoHeader res = new BitmapInfoHeader();
            res.bitsPerPixel = 8;
            res.colorPlanes = 1;
            res.compression = BitmapCompression.BI_RGB;
            res.cwSize = 40;
            res.width = width;
            res.height = height;
            res.hrez = 10000;
            res.vrez = 10000;
            res.imageSize = width * height;
            res.numberOfColorsInPalette = 256;
            res.numberOfImportantColors = 0;

            return res;
        }
    }

    public class GrayBitmap
    {
        uint width;
        uint height;
        byte[] data;

        public uint Width { get { return width; } }
        public uint Height { get { return width; } }

        public byte this[uint x, uint y] { get { return data[x +  y * width]; } set { data[x + y * width] = value; } }

        /// actually creates a copy
        public ushort[] PixelsUShort
        {
            get { return data.Select(i => { return (ushort)i; }).ToArray(); }
            set { data = value.Select(i => { return (byte)i; }).ToArray(); }
        }

        public GrayBitmap(uint width, uint height)
        {
            this.width = width;
            this.height = height;
            data = new byte[width * height];
        }

        public static GrayBitmap Load(string filename)
        {
            using (BinaryReader br = new BinaryReader(new FileStream(filename, FileMode.Open)))
            {
                byte[] header = br.ReadBytes(54);
                GCHandle gch = GCHandle.Alloc(header, GCHandleType.Pinned);

                // read headers
                FileHeader fh = (FileHeader)Marshal.PtrToStructure(Marshal.UnsafeAddrOfPinnedArrayElement(header, 0), typeof(FileHeader));
                BitmapInfoHeader bih = (BitmapInfoHeader)Marshal.PtrToStructure(Marshal.UnsafeAddrOfPinnedArrayElement(header, 14), typeof(BitmapInfoHeader));

                // read the palette
                br.ReadBytes((int)(bih.numberOfColorsInPalette * 4));

                //Console.Out.WriteLine("Loading bitmap file width = {0} - height = {1}", bih.width, bih.height) ;
                
                GrayBitmap res = new GrayBitmap(bih.width, bih.height);

                // read data - note that bmp starts with bottom-left corner
                for (int y = 0; y < res.height; ++y)
                    br.Read(res.data, (int)res.width *((int)res.height - 1 - y), (int)res.width);
                return res;
            }
        }

        public void Save(string filename)
        {
            using (BinaryWriter bw = new BinaryWriter(new FileStream(filename, FileMode.OpenOrCreate)))
            {
                // write headers
                FileHeader header = FileHeader.GrayBitmap(width, height);
                BitmapInfoHeader bih = BitmapInfoHeader.GrayBitmap(width, height);

                byte[] head = new byte[54];

                GCHandle gch = GCHandle.Alloc(header, GCHandleType.Pinned);
                Marshal.Copy(gch.AddrOfPinnedObject(), head, 0, 14);
                gch.Free();

                gch = GCHandle.Alloc(bih, GCHandleType.Pinned);
                Marshal.Copy(gch.AddrOfPinnedObject(), head, 14, 40);
                gch.Free();

                bw.Write(head, 0, 54);

                // write color table
                uint[] colors = new uint[256];
                uint color = 0;
                uint next = 0x010101;
                for (int k = 0; k < 256; ++k)
                {
                    bw.Write(color);
                    color += next;
                }

                // write data - note that bmp starts with bottom-left corner
                for (int y = 0; y < height; ++y)
                    bw.Write(data, (int)width * ((int)height - 1 - y), (int)width);
            }
        }
    }
}