from PyQt5 import QtWidgets
from pyqtgraph.Qt import QtCore
import pyqtgraph.opengl as gl
from pyqtgraph.opengl.shaders import VertexShader, FragmentShader, ShaderProgram
import numpy as np

from Geometry_Helper import IcosahedronSphere

ShaderProgram('translation_in_tunnel', [
    VertexShader("""    
    
    // Constants
    const float c_pi = 3.14159265359;

    // Uniforms
    uniform float u_time;
    uniform float u_velocity;

    // Variables
    varying float v_distance;
    varying vec3 v_point_in_plane;

    void main() {
    
        // Fish position
        vec3 fish_pos = vec3(0.0, 0.0, 0.0);

        // Absolute vertex position
        vec3 position = gl_Vertex.xyz;

        // Set plane of tunnel wall segment
        vec3 plane_normal = normalize(vec3(0.0, position.yz));
        vec3 plane_offset = normalize(vec3(0.0, position.yz));
        
        // Calculate scalar factor
        float scale_factor = dot(plane_offset - fish_pos, plane_normal) / dot(position, plane_normal);
        
        // Calculate point in plane (tunnel wall)
        v_point_in_plane = scale_factor * position + fish_pos;
        
        // Calculate distance traveled
        v_distance = u_time * u_velocity;
        
        // Set display vertex position
        gl_Position = ftransform();
    }
    """),

    FragmentShader("""

    // Constants
    const float c_pi = 3.14159265359;
    const float c_thresh = 15.0;

    // Uniforms
    uniform float u_time;
    uniform float u_sf;

    // Variables
    varying float v_distance;
    varying vec3 v_point_in_plane;

    void main() {
        
        float texc = v_point_in_plane.x;
     
        // Set luminance value
        float newcolor1 = (1 + sin((texc + v_distance) * 2.0 * c_pi * u_sf * 180)) / 2.0;
        float newcolor2 = (1 + sin((texc + v_distance) * 2.0 * c_pi * u_sf * 180 + c_pi)) / 2.0;
        
        // Add distance fade effect
        newcolor1 = newcolor1 * pow(1.0 * cos(0.1 * texc), 2);
        //newcolor1 /= floor((5.0 + u_time) / 5.0);
        
        newcolor2 = newcolor2 * pow(1.0 * cos(0.1 * texc), 2);
        //newcolor2 /= floor((5.0 + u_time) / 5.0);
        
        // Blank front and back to prevent aliasing artifacts
        if (texc > c_thresh || texc < -c_thresh) {
            newcolor1 = 0.0;
            newcolor2 = 0.0;
        }
        
        float scale = 0.0;
        if (u_time > 5.0) {
            scale = (u_time-5.0) / 30;
        }
        
        // Set fragment color
        gl_FragColor = vec4(newcolor1, scale * newcolor2, 0.0,  1.0);
    }
    """)
]),

u_time = None
next_phase = None
next_phase_time = None
camera_dist = 0.01
camera_elev = 0.


def init_shader():
    global u_time, mi
    u_time = 0.

    mi.shader().setUniformData('u_velocity', [1.0])
    mi.shader().setUniformData('u_sf', [0.005])

    mi.shader().setUniformData('u_time', [0.0])
    mi.update()


# Define update function
def update():
    global u_time, camera_dist, camera_elev, start_zoomout, mi, frametime

    u_time += frametime

    mi.shader().setUniformData('u_time', [u_time])
    mi.update()

if __name__ == '__main__':

    frametime = 1./60

    # Create sphere
    sphere = IcosahedronSphere(subdiv_lvl=6)
    verts = sphere.getVertices()
    faces = sphere.getFaces()

    # Set MeshData
    md = gl.MeshData()
    md.setVertexes(verts)
    md.setFaces(faces)

    # Setup app and window
    app = QtWidgets.QApplication([])
    window = gl.GLViewWidget()
    #window.resize(QtCore.QSize(1000, 1000))
    window.
    window.showFullScreen()
    window.show()
    window.setWindowTitle('Translation in tunnel')
    window.setCameraPosition(elevation=camera_elev, distance=camera_dist, azimuth=180)

    # Create MeshItem
    gi = gl.GLGridItem()
    window.addItem(gi)
    mi = gl.GLMeshItem(meshdata=md, smooth=True, shader='translation_in_tunnel')
    window.addItem(mi)
    init_shader()


    # Set timer for frame update
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(1000 * frametime)

    # Start event loop
    QtWidgets.QApplication.instance().exec_()