from PyQt5 import QtWidgets
from pyqtgraph.Qt import QtCore
import pyqtgraph.opengl as gl
from pyqtgraph.opengl.shaders import VertexShader, FragmentShader, ShaderProgram
import numpy as np

from Geometry_Helper import UVCylinder

ShaderProgram('transl_rot_shader', [
    VertexShader("""

    mat4 rotationY( in float angle ) {
        return mat4(
                    cos(angle),	    0.0,    sin(angle),	    0.0,
                    0.0,		    1.0,	0.0,	        0.0,
                    -sin(angle),	0.0,	cos(angle),	    0.0,
                    0.0, 		    0.0,	0.0,	        1.0
                    );
    }


    const float c_pi = 3.14159265359;
    
    uniform float u_time;
    uniform float  u_fish_pos_x;
    uniform float  u_fish_pos_y;

    varying vec3 v_pos;

    void main() {

        // Absolute vertex position
        v_pos = (rotationY(c_pi/2.0) * gl_Vertex).xyz;
        
        // Display vertex position
        vec4 position = ftransform();
        position.x += u_fish_pos_x;
        position.y += u_fish_pos_y;
        gl_Position = position;
    }
    """),

    FragmentShader("""
    
    mat4 rotationX( in float angle ) {
        return mat4(
                    1.0,	0.0,		0.0,		    0.0,
                    0.0, 	cos(angle),	-sin(angle),	0.0,
                    0.0, 	sin(angle), cos(angle),		0.0,
                    0.0, 	0.0,		0.0, 		    1.0
                    );
    }

    mat4 rotationY( in float angle ) {
        return mat4(
                    cos(angle),	    0.0,    sin(angle),	    0.0,
                    0.0,		    1.0,	0.0,	        0.0,
                    -sin(angle),	0.0,	cos(angle),	    0.0,
                    0.0, 		    0.0,	0.0,	        1.0
                    );
    }

    mat4 rotationZ( in float angle ) {
        return mat4(
                    cos(angle),	-sin(angle),	0.0,	0.0,
                    sin(angle),	cos(angle),		0.0,	0.0,
                    0.0,	    0.0,	        1.0,	0.0,
                    0.0,	    0.0,            0.0,    1.0
                    );
    }
    
    const float c_pi = 3.14159265359;
    const float c_thresh = 15.0;

    uniform float u_time;
    uniform float u_velocity;
    uniform float u_sf;

    varying vec3 v_pos;

    void main() {
    
        vec3 newcolor;
        
        newcolor = sin((v_pos.x + u_velocity * u_time) * 2.0 * c_pi * u_sf * 180);
    
        // Add distance fade effect
        newcolor = newcolor * pow(1.0 * cos(1.0/8.0 * v_pos.x), 2);
        
        if (v_pos.x > 20.0 | v_pos.x < -20.0) {
            newcolor = vec3(0.0, 0.0, 0.0);
        }
        
        // Set fragment color
        gl_FragColor = vec4(newcolor.xyz, 1.0);
    }
    """)
]),

u_time = None
next_phase = None
next_phase_time = None

def currentFishPos(time):
    return [0.0, 0.95 * np.sin(0.1 * 2.0 * np.pi * time)]
    #return [0.5 * np.sin(0.3 * 2.0 * np.pi * time), 0.8 * np.cos(0.3 * 2.0 * np.pi * time)]

def init_shader():
    global u_time, mi
    u_time = 0.

    mi.shader().setUniformData('u_velocity', [1.5])
    mi.shader().setUniformData('u_sf', [0.01])
    mi.shader().setUniformData('u_fish_pos_x', [currentFishPos(u_time)[0]])
    mi.shader().setUniformData('u_fish_pos_y', [currentFishPos(u_time)[1]])


    mi.shader().setUniformData('u_time', [0.0])
    mi.update()


# Define update function
def update():
    global u_time, mi, window, frametime

    u_time += frametime

    window.setCameraPosition(elevation=90 + 10.0 * np.cos(0.1 * 2.0 * np.pi * u_time))

    mi.shader().setUniformData('u_fish_pos_x', [currentFishPos(u_time)[0]])
    mi.shader().setUniformData('u_fish_pos_y', [currentFishPos(u_time)[1]])
    mi.shader().setUniformData('u_time', [u_time])
    mi.update()

if __name__ == '__main__':

    frametime = 1./60

    # Create cylinder
    cylinder = UVCylinder(60, 60, height=40)
    verts = cylinder.getVertices()
    faces = cylinder.getFaceIndices()

    # Set MeshData
    md = gl.MeshData()
    md.setVertexes(verts)
    md.setFaces(faces)

    # Setup app and window
    app = QtWidgets.QApplication([])
    window = gl.GLViewWidget()
    window.resize(QtCore.QSize(1000, 900))
    window.show()
    window.setWindowTitle('Stimulus preview')
    window.setCameraPosition(elevation=90, distance=0.01, azimuth=0)

    # Create MeshItem
    gi = gl.GLGridItem()
    window.addItem(gi)
    mi = gl.GLMeshItem(meshdata=md, smooth=True, shader='transl_rot_shader')
    window.addItem(mi)
    init_shader()


    # Set timer for frame update
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(1000 * frametime)

    # Start event loop
    QtWidgets.QApplication.instance().exec_()