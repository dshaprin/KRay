#include "init.h"
#include "Scene.h"
#include "camera.h"
int windowWidth = 640, windowHeight= 480;
GLuint  bufferObj;
cudaGraphicsResource *resource;
uchar4* devPtr;
Camera camera;
Scene* scene;
void keyFunc( unsigned char key, int x, int y ) {
    switch (key) {
		case 's':
			camera.rotate(0,1);
		break;
		case 'w':
			camera.rotate(0,-1);
		break;
		case 'a':
			camera.rotate(1,0);
		break;
		case 'd':
			camera.rotate(-1,0);
		break;
		case 'i':
			camera.move(0,1);
		break;	
		case 'k':
			camera.move(0,-1);
		break;
		case 'j':
			camera.move(-1,0);
		break;
		case 'l':
			camera.move(1,0);
		break;
        case 27:
            HANDLE_ERROR( cudaGraphicsUnregisterResource( resource ) );
            glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
            glDeleteBuffers( 1, &bufferObj );
			delete scene;
            exit(0);
    }
}

void update()
{
	  // do work with the memory dst being on the GPU, gotten via mapping
	render(devPtr,resource, camera, glutGet(GLUT_WINDOW_WIDTH),glutGet(GLUT_WINDOW_HEIGHT));
}
static int frame=0,t,timebase=0; 
static float fps;
void windowReshapeFunc( GLint newWidth, GLint newHeight )
{
	windowWidth = newWidth;
	windowHeight = newHeight;
	glViewport(0, 0, newWidth, newHeight);
	HANDLE_ERROR( cudaGraphicsUnregisterResource( resource ) );
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
    glDeleteBuffers( 1, &bufferObj );

	glGenBuffers( 1, &bufferObj );
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj );
	glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, windowWidth * windowHeight* 4,
                  NULL, GL_DYNAMIC_DRAW_ARB );
	 HANDLE_ERROR( 
        cudaGraphicsGLRegisterBuffer( &resource, 
                                      bufferObj, 
                                      cudaGraphicsMapFlagsNone ) );
}
void drawFunc( void ) 
{
	frame++;
	t=glutGet(GLUT_ELAPSED_TIME);

	if (t - timebase > 1000) {
		fps = frame*1000.0/(t-timebase);
	 	timebase = t;
		frame = 0;
		printf("FPS:%2.2f\n",fps);
	};
	update();
	glDrawPixels( glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT), GL_RGBA, GL_UNSIGNED_BYTE, 0 );
	glutSwapBuffers();
}
void initCamera()
{
	camera.init();
	camera.pos = Vector(0.0f, -10.0f, 0.0f);
	camera.yaw = 0.0;
	camera.pitch = 0.0;
	camera.roll = 0.0;
	camera.fov = 60;
	camera.aspect = 4.0/3.0f;

}

void init(int argc, char* argv[])
{
	initCamera();
	initColor();
	cudaDeviceProp  prop;
    int dev;
    memset( &prop, 0, sizeof( cudaDeviceProp ) );
    prop.major = 2;
    prop.minor = 0;
    HANDLE_ERROR( cudaChooseDevice( &dev, &prop ) );


    // these GLUT calls need to be made before the other OpenGL
    // calls, else we get a seg fault
    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
	glutInitWindowSize( windowWidth, windowHeight);
    glutCreateWindow( TITLE );

    glBindBuffer    = (PFNGLBINDBUFFERARBPROC)GET_PROC_ADDRESS("glBindBuffer");
    glDeleteBuffers = (PFNGLDELETEBUFFERSARBPROC)GET_PROC_ADDRESS("glDeleteBuffers");
    glGenBuffers    = (PFNGLGENBUFFERSARBPROC)GET_PROC_ADDRESS("glGenBuffers");
    glBufferData    = (PFNGLBUFFERDATAARBPROC)GET_PROC_ADDRESS("glBufferData");

    // the first three are standard OpenGL, the 4th is the CUDA reg 
    // of the bitmap these calls exist starting in OpenGL 1.5
    glGenBuffers( 1, &bufferObj );
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj );
	glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, windowWidth * windowHeight* 4,
                  NULL, GL_DYNAMIC_DRAW_ARB );

    HANDLE_ERROR( 
        cudaGraphicsGLRegisterBuffer( &resource, 
                                      bufferObj, 
                                      cudaGraphicsMapFlagsNone ) );
	scene = new Scene();
}
