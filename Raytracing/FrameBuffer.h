#include "cuda.h"
#include "vector.h"
#include "color.h"
struct FrameBufferElement
{
	Color color;
	Ray nextRay;

};
class FrameBuffer
{
	FrameBufferElement* data;
	int width, height;
	void destroy();
	void create();
	void create(int w, int h);
	void copy(const FrameBuffer& f);
public:
	FrameBuffer(int w, int h)
		:width(w),height(h)
	{
		create();
	}
	FrameBuffer(const FrameBuffer& f)
	{
		copy(f);
	}
	~FrameBuffer()
	{
		destroy();
	}
	FrameBuffer& operator=(const FrameBuffer& f);
	const FrameBufferElement* getData()const
	{
		return data;
	}
};
