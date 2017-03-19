extern "C"
{
	__global__ void Sobel(const unsigned char* input, unsigned int* output, const unsigned int width, const unsigned int height)
	{
		const int col = blockIdx.x * blockDim.x + threadIdx.x;
		const int row = blockIdx.y * blockDim.y + threadIdx.y;
		
		if (col > width-1 || row > height-1 || row == 0 || col == 0)
		{
			return;
		}

		const int index = col + row * width;
		double dx = input[index-width+1] + 2 * input[index+1] + input[index+width+1] - input[index-width-1] - 2 * input[index-1] - input[index+width-1];
		double dy = input[index+width-1] + 2 * input[index+width] + input[index+width+1] - input[index-width-1] - 2 * input[index-width] - input[index-width+1];		
		int magnitude = (int)( (sqrt(dx * dx + dy * dy)*255)/1141 );

		output[index] = magnitude;
	}
}