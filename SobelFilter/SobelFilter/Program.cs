using System;
using ManagedCuda;
using ManagedCuda.VectorTypes;
using System.Drawing;
using System.Drawing.Imaging;
using System.Threading.Tasks;
using System.Diagnostics;

namespace SobelFilter
{
    class Program
    {
        unsafe static void Main(string[] args)
        {
            const string inputPath = @"C:\Users\t-fegasp\Pictures\kitten.jpg";
            const string outputPath = @"C:\Users\t-fegasp\Pictures\kitten.out.jpg";

            Bitmap image = loadImageTo32bppFormat(inputPath);
            byte[] grayData = ConvertTo8bpp(image);
            Int64 sTicks = singleThreadedSobel(image, grayData);
            // Int64 sTicks = GPUSobel(image, grayData);
            Int64 pTicks = parallelSobel(image, grayData);
            Console.WriteLine($"Single took: {sTicks} ticks. Parallel took: {pTicks} ticks. Speedup was {(double)sTicks / (double)pTicks}.");
            image.Save(outputPath);
            Console.ReadKey();
        }

        unsafe static public Int64 GPUSobel(Bitmap image, byte[] grayData)
        {
            int width = image.Width;
            int height = image.Height;

            var context = new CudaContext();
            dim3 blockDim = new dim3(16, 16);
            uint gridX = (uint)(width + blockDim.x - 1) / blockDim.x;
            uint gridY = (uint)(height + blockDim.y - 1) / blockDim.y;
            dim3 gridDim = new dim3(gridX, gridY);
            CudaKernel kernel = context.LoadKernelPTX("Kernel.ptx", "Sobel");
            kernel.BlockDimensions = blockDim;
            kernel.GridDimensions = gridDim;

            BitmapData imageData = image.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadWrite, image.PixelFormat);
            uint* ptr = (uint*)imageData.Scan0.ToPointer(); // An unsigned int pointer. This points to the image data in memory, each uint is one pixel ARGB
            int stride = imageData.Stride / 4; // Stride is the width of one pixel row, including any padding. In bytes, /4 converts to 4 byte pixels
            CudaDeviceVariable<byte> deviceGrayData = grayData;
            CudaDeviceVariable<uint> output = new CudaDeviceVariable<uint>(width * height);
            Stopwatch sw = Stopwatch.StartNew();
            kernel.Run(deviceGrayData.DevicePointer, output.DevicePointer, width, height);
            sw.Stop();
            Int64 ticks = sw.ElapsedTicks;

            uint[] filteredImage = output;
            int index = 0;
            for (int i = 1; i < height; ++i)
            {
                for (int j = 1; j < width; ++j)
                {
                    *(ptr + i * stride + j) = filteredImage[index++];
                }
            }
            for (int x = 0; x < width; ++x)
            {
                *(ptr + (height - 1) * stride + x) = 0;
                *(ptr + x) = 0;
            }
            for (int y = 0; y < height; ++y)
            {
                *(ptr + y * stride) = 0;
                *(ptr + y * stride + width - 1) = 0;
            }
            // Finish with image and save
            image.UnlockBits(imageData);

            return ticks;
        }

        unsafe static public Int64 parallelSobel(Bitmap image, byte[] grayData)
        {
            int width = image.Width;
            int height = image.Height;

            BitmapData imageData = image.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadWrite, image.PixelFormat);

            uint* ptr = (uint*)imageData.Scan0.ToPointer();
            int stride = imageData.Stride / 4;

            int processorCount = Environment.ProcessorCount;
            int rowChunkSize = (height + processorCount - 1) / processorCount;
            Console.WriteLine($"We have {processorCount} processors. The width is {width} and the height is {height}. So the rowChunkSize is {rowChunkSize}.");
            Stopwatch sw = Stopwatch.StartNew();
            Parallel.For(0, Environment.ProcessorCount, i =>
            {
                int rowsFrom = i * rowChunkSize;
                if (rowsFrom == 0)
                {
                    rowsFrom = 1;
                }
                int rowsTo = (i + 1) * rowChunkSize;
                if (rowsTo > height - 1)
                {
                    rowsTo = height - 1;
                }
                byte[] buffer = new byte[9];
                for (int y = rowsFrom; y < rowsTo; ++y)
                {
                    for (int x = 1; x < width - 1; ++x)
                    {
                        int index = y * width + x;
                        buffer[0] = grayData[index - width - 1];
                        buffer[1] = grayData[index - width];
                        buffer[2] = grayData[index - width + 1];
                        buffer[3] = grayData[index - 1];
                        buffer[4] = grayData[index];
                        buffer[5] = grayData[index + 1];
                        buffer[6] = grayData[index + width - 1];
                        buffer[7] = grayData[index + width];
                        buffer[8] = grayData[index + width + 1];

                        double dx = buffer[2] + 2 * buffer[5] + buffer[8] - buffer[0] - 2 * buffer[3] - buffer[6];
                        double dy = buffer[6] + 2 * buffer[7] + buffer[8] - buffer[0] - 2 * buffer[1] - buffer[2];
                        double magnitude = Math.Sqrt(dx * dx + dy * dy) / 1141; // 1141 is approximately the max sobel response
                        byte grayMag = Convert.ToByte(magnitude * 255);
                        *(ptr + y * stride + x) = (0xFF000000 | (uint)(grayMag << 16) | (uint)(grayMag << 8) | grayMag);
                    }
                }
            });
            sw.Stop();
            Int64 ticks = sw.ElapsedTicks;

            for (int x = 0; x < width; ++x)
            {
                *(ptr + (height - 1) * stride + x) = 0;
                *(ptr + x) = 0;
            }
            for (int y = 0; y < height; ++y)
            {
                *(ptr + y * stride) = 0;
                *(ptr + y * stride + width - 1) = 0;
            }
            // Release image
            image.UnlockBits(imageData);

            return ticks;
        }

        unsafe static public Int64 singleThreadedSobel(Bitmap image, byte[] grayData)
        {
            int width = image.Width;
            int height = image.Height;

            byte[] buffer = new byte[9];
            BitmapData imageData = image.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadWrite, image.PixelFormat);
            uint* ptr = (uint*)imageData.Scan0.ToPointer();
            int stride = imageData.Stride / 4;
            Stopwatch sw = Stopwatch.StartNew();
            for (int y = 1; y < height - 1; y++)
            {
                for (int x = 1; x < width - 1; x++)
                {
                    int index = y * width + x;
                    buffer[0] = grayData[index - width - 1];
                    buffer[1] = grayData[index - width];
                    buffer[2] = grayData[index - width + 1];
                    buffer[3] = grayData[index - 1];
                    buffer[4] = grayData[index];
                    buffer[5] = grayData[index + 1];
                    buffer[6] = grayData[index + width - 1];
                    buffer[7] = grayData[index + width];
                    buffer[8] = grayData[index + width + 1];
                    // Sobel horizontal and vertical
                    double dx = buffer[2] + 2 * buffer[5] + buffer[8] - buffer[0] - 2 * buffer[3] - buffer[6];
                    double dy = buffer[6] + 2 * buffer[7] + buffer[8] - buffer[0] - 2 * buffer[1] - buffer[2];
                    double magnitude = Math.Sqrt(dx * dx + dy * dy) / 1141; // 1141 is approximately the max sobel response
                    byte grayMag = Convert.ToByte(magnitude * 255);
                    *(ptr + y * stride + x) = (0xFF000000 | (uint)(grayMag << 16) | (uint)(grayMag << 8) | grayMag);
                }
            }
            sw.Stop();
            Int64 ticks = sw.ElapsedTicks;

            for (int x = 0; x < width; ++x)
            {
                *(ptr + (height - 1) * stride + x) = 0;
                *(ptr + x) = 0;
            }
            for (int y = 0; y < height; ++y)
            {
                *(ptr + y * stride) = 0;
                *(ptr + y * stride + width - 1) = 0;
            }
            // Release lock
            image.UnlockBits(imageData);

            return ticks;
        }

        public static Bitmap loadImageTo32bppFormat(string inputPath)
        {
            Bitmap image = new Bitmap(inputPath);

            if (image.PixelFormat != PixelFormat.Format32bppArgb
                && image.PixelFormat != PixelFormat.Format32bppRgb)
            {
                image = image.Clone(new Rectangle(0, 0, image.Width, image.Height), PixelFormat.Format32bppArgb);
            }

            return image;
        }

        unsafe public static byte[] ConvertTo8bpp(Bitmap image)
        {
            int width = image.Width;
            int height = image.Height;

            byte[] grayData = new byte[width * height];

            BitmapData imageData = image.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadOnly, image.PixelFormat);

            uint* ptr = (uint*)imageData.Scan0.ToPointer();
            int inputStride = imageData.Stride / 4;

            byte r, g, b;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    uint pixel = *(ptr + y * inputStride + x);
                    r = (byte)((0x00FF0000 & pixel) >> 16);
                    g = (byte)((0x0000FF00 & pixel) >> 8);
                    b = (byte)((0x000000FF & pixel));

                    byte gray = (byte)(0.2126 * r + 0.7152 * g + 0.0722 * b);
                    grayData[y * width + x] = gray;
                }
            }

            image.UnlockBits(imageData);

            return grayData;
        }
    }
}
