from PyQt5 import QtWidgets
from pyqtgraph.Qt import QtCore
import pyqtgraph.opengl as gl
from pyqtgraph.opengl.shaders import VertexShader, FragmentShader, ShaderProgram
import numpy as np

from Geometry_Helper import IcosahedronSphere

ShaderProgram('conj_trans_rot_through_tunnel', [
    VertexShader("""    
    
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
    // Constants
    const float c_pi = 3.14159265359;

    // Uniforms
    uniform float u_time;
    uniform float u_velocity;

    // Variables
    varying float v_distance;
    varying vec3 v_point_in_plane;

    void main() {

        // Absolute vertex position
        vec4 vertex = gl_Vertex;
        
        float u_yaw_amplitude = 0.8;
        float u_yaw_rot_period = 0.1;
        float u_rot_rate_pitch = 0.0;
        
        // Set fish position (do this outside shader in future)
        vec3 u_fish_pos = vec3(0.0, 
                               u_yaw_amplitude * sin(u_yaw_rot_period * 2.0 * c_pi * u_time), 
                               0.0
                               );
        
        // Apply pitch and yaw rotations to v_pos
        float yaw_angle = c_pi / 4.0 * u_yaw_amplitude / u_velocity * -cos(u_yaw_rot_period * 2.0 * c_pi * u_time);
        //vertex = rotationZ(yaw_angle) * vertex;
        
        // Set final position
        vec3 position = vertex.xyz;

        // Set plane of tunnel wall segment
        vec3 plane_normal = normalize(vec3(0.0, position.yz));
        vec3 plane_offset = normalize(vec3(0.0, position.yz));
        
        // Calculate scalar factor
        float scale_factor = dot(plane_offset - u_fish_pos, plane_normal) / dot(position, plane_normal);
        
        // Calculate point in plane (tunnel wall)
        v_point_in_plane = scale_factor * position + u_fish_pos;
        
        // Calculate distance traveled
        v_distance = u_time * u_velocity; //+ u_velocity * 0.3 * (1-cos(u_time * 4.0 * c_pi * u_yaw_rot_period));
        
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
     
        float newcolor = sin((texc + v_distance) * 2.0 * c_pi * u_sf * 180);
        
        // Add distance fade effect
        newcolor = newcolor * pow(1.0 * cos(0.1 * texc), 2);
        
        // Blank front and back to prevent aliasing artifacts
        if (texc > c_thresh || texc < -c_thresh) {
            newcolor = 0.0;
        }
        
        // Set fragment color
        gl_FragColor = vec4(newcolor, newcolor, newcolor, 1.0);
    }
    """)
]),

u_time = None
next_phase = None
next_phase_time = None
camera_dist = 0.01
camera_elev = 1.

start_zoomout = np.inf

def init_shader():
    global u_time, mi
    u_time = 0.

    mi.shader().setUniformData('u_velocity', [2.0])
    mi.shader().setUniformData('u_sf', [0.005])

    mi.shader().setUniformData('u_time', [0.0])
    mi.update()


# Define update function
def update():
    global u_time, camera_dist, camera_elev, start_zoomout, mi, frametime

    u_time += frametime

    if u_time > start_zoomout:
        camera_dist = 0.3 * (u_time - start_zoomout)
        camera_dist = 3. if camera_dist >= 3. else camera_dist

        camera_elev = 3 * (u_time - start_zoomout)
        camera_elev = 70. if camera_elev >= 70. else camera_elev

    #window.setCameraPosition(distance=camera_dist, elevation=camera_elev)

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
    window.setWindowTitle('Stimulus preview')
    window.setCameraPosition(elevation=camera_elev, distance=camera_dist, azimuth=180)

    # Create MeshItem
    gi = gl.GLGridItem()
    window.addItem(gi)
    mi = gl.GLMeshItem(meshdata=md, smooth=True, shader='conj_trans_rot_through_tunnel')
    window.addItem(mi)
    init_shader()


    # Set timer for frame update
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(1000 * frametime)

    # Start event loop
    QtWidgets.QApplication.instance().exec_()