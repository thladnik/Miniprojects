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
    uniform float u_fish_pos_x;
    uniform float u_fish_pos_y;
    uniform float u_fish_pos_z;

    // Variables
    varying float v_distance;
    varying vec3 v_point_in_plane;
    varying vec3 v_pos;

    void main() {
    
        // Fish position
        vec3 fish_pos = vec3(u_fish_pos_x, u_fish_pos_y, u_fish_pos_z);

        // Absolute vertex position
        v_pos = gl_Vertex.xyz;
        vec3 position = v_pos;

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
    uniform float u_ripple_sf_factor;

    // Variables
    varying float v_distance;
    varying vec3 v_point_in_plane;
    varying vec3 v_pos;

    void main() {
        
        float texc = v_point_in_plane.x;
     
        // Set luminance value
        float newcolor = (1 + sin((texc + v_distance) * 2.0 * c_pi * u_sf * 180)) /  2.0;
        
        newcolor += newcolor * 0.05 * abs(v_pos.z - 1) * cos((v_pos.y + 0.5 * u_time) * 2.0 * c_pi * 10.0);
        
        // Add distance fade effect
        newcolor = newcolor * pow(1.0 * cos(0.1 * texc), 2);
        //newcolor /= floor((5.0 + u_time) / 5.0);
        
        // Blank front and back to prevent aliasing artifacts
        if (texc > c_thresh || texc < -c_thresh) {
            newcolor = 0.0;
        }
        
        // Set fragment color
        gl_FragColor = vec4(newcolor, newcolor, newcolor,  1.0);
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
    mi.shader().setUniformData('u_fish_pos_x', [0.0])
    mi.shader().setUniformData('u_fish_pos_y', [0.0])
    mi.shader().setUniformData('u_fish_pos_z', [0.0])
    #mi.shader().setUniformData('u_ripple_sf_factor', [1.0])

    mi.shader().setUniformData('u_time', [0.0])
    mi.update()


# Define update function
def update():
    global u_time, camera_dist, camera_elev, start_zoomout, mi, frametime

    u_time += frametime

    if False:
        if u_time > 10.:
            mi.shader().setUniformData('u_ripple_sf_factor', [0.5])
        if u_time > 20.:
            mi.shader().setUniformData('u_ripple_sf_factor', [1.0])
        if u_time > 30.:
            mi.shader().setUniformData('u_ripple_sf_factor', [2.0])
        if u_time > 40.:
            mi.shader().setUniformData('u_ripple_sf_factor', [3.0])

    #mi.shader().setUniformData('u_fish_pos_y', [0.2 * np.sin(2. * np.pi * u_time * 0.5)])
    #mi.shader().setUniformData('u_fish_pos_z', [0.3 * np.cos(2. * np.pi * u_time * 0.3)])

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
    window.resize(QtCore.QSize(1000, 1000))
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