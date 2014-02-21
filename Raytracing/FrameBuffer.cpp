#include "FrameBuffer.h"
void FrameBuffer::destroy()
{
	delete[] data;
}
void FrameBuffer::create()
{
	data = new FrameBufferElement[width * height];
}
void FrameBuffer::create(int x, int y)
{
	width = x; height = y;
	create();
}
void FrameBuffer::copy(const FrameBuffer& f)
{
	create(f.width,f.height);
	for(int i = 0; i < width * height ; ++i)
		f.data[i] = data[i];
}