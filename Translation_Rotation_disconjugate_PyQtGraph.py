from PyQt5 import QtWidgets
from pyqtgraph.Qt import QtCore
import pyqtgraph.opengl as gl
from pyqtgraph.opengl.shaders import VertexShader, FragmentShader, ShaderProgram
import numpy as np

from Geometry_Helper import IcosahedronSphere

ShaderProgram('transl_rot_shader', [
    VertexShader("""

    varying vec4 v_pos;

    void main() {

        // Absolute vertex position
        v_pos = gl_Vertex;
        // Display vertex position
        gl_Position = ftransform();
    }
    """),

    FragmentShader("""

    float rand(vec2 co){
        return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
    }

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
    uniform float u_rot_rate_pitch;
    uniform float u_rot_rate_yaw;
    uniform float u_start_time_phase;
    uniform bool u_use_harmonic01;
    uniform bool u_use_harmonic02;

    varying vec4 v_pos;

    void main() {
        
        float rel_time = u_time - u_start_time_phase;
        
        // Rotate
        v_pos = rotationY(-u_rot_rate_pitch*2.0*c_pi*rel_time) * rotationZ(u_rot_rate_yaw*2.0*c_pi*rel_time) * v_pos;

        vec3 newcolor;
        //vec3 color = vec3(0.0, 0.0, 0.0);

        // Calculate coordinate on cylinder wall
        float texc = tan(asin(v_pos.x));

        // Blank front and back to prevent aliasing artifacts
        if (texc > c_thresh || texc < -c_thresh) {
            gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
            return;
            
        // Set color otherwise
        } else {
            newcolor = sin((u_velocity * u_time + texc) * u_sf);
            
            if (u_use_harmonic01) {
                newcolor = newcolor * sin((u_velocity * u_time + texc) * 2.0 * u_sf);
            }
            if (u_use_harmonic02) {
                newcolor = newcolor * sin((u_velocity * u_time + texc) * 3.0 * u_sf);
            }
            
            // Add distance fade effect
            newcolor = newcolor * pow(3.0 * cos(asin(v_pos.x)), 2);
            
            // Set fragment color
            gl_FragColor = vec4(newcolor.xyz, 1.0);
        }
    }
    """)
]),

u_time = None
next_phase = None
next_phase_time = None

def init_shader():
    global u_time, start_delay, start_phase, next_phase, next_phase_time
    u_time = -start_delay
    next_phase = start_phase
    next_phase_time = phase_transl_time + phase_delay

    mi.shader().setUniformData('u_rot_rate_pitch', [0.0])
    mi.shader().setUniformData('u_rot_rate_yaw', [0.0])
    mi.shader().setUniformData('u_velocity', [0.0])
    mi.shader().setUniformData('u_sf', [5.0])
    mi.shader().setUniformData('u_start_time_phase', [0.])
    mi.shader().setUniformData('u_use_harmonic01', [False])
    mi.shader().setUniformData('u_use_harmonic02', [False])

    mi.shader().setUniformData('u_time', [0.0])
    mi.update()


# Define update function
shader_changed = True
def update():
    global u_time, mi, window, next_phase, next_phase_time, elev_offset

    u_time += frametime

    # Update camera
    set_camera = False
    if u_time >= phase_transl_time:
        new_elev = elev_offset
        new_azim = start_azimuth + 180.
        new_dist = 0.01
        set_camera = True

    elif u_time >= 0.:
        # Initial "fly-in" of camera
        new_elev = start_elev - u_time * start_elev / phase_transl_time + elev_offset
        new_azim = start_azimuth + 180 * np.square((1 - np.cos(np.pi * u_time/phase_transl_time)) / 2)
        new_dist = start_distance - start_distance * np.power(u_time / phase_transl_time, 12)
        set_camera = True

    if set_camera:
        window.setCameraPosition(elevation=new_elev, distance=new_dist, azimuth=new_azim)

    #####
    ## Phase switches
    if u_time > next_phase_time and next_phase == 'transl':
        print('Switch to phase TRANSLATION at time %.1f' % u_time)
        next_phase = 'rot_pitch'
        next_phase_time += phase_transl_time + phase_delay

    elif u_time > next_phase_time and next_phase == 'rot_pitch':
        print('Switch to phase PITCH at time %.1f' % u_time)
        mi.shader().setUniformData('u_rot_rate_pitch', [phase_rot_pitch_rot_rate])
        mi.shader().setUniformData('u_start_time_phase', [next_phase_time])
        next_phase = 'rot_yaw'
        next_phase_time += 1./phase_rot_pitch_rot_rate + phase_delay

    elif u_time > next_phase_time and next_phase == 'rot_yaw':
        print('Switch to phase YAW at time %.1f' % u_time)
        mi.shader().setUniformData('u_rot_rate_yaw', [phase_rot_yaw_rot_rate])
        mi.shader().setUniformData('u_start_time_phase', [next_phase_time])
        next_phase = 'rot_combined'
        next_phase_time += 1./phase_rot_yaw_rot_rate + phase_delay

    elif u_time > next_phase_time and next_phase == 'rot_combined':
        print('Switch to phase YAW at time %.1f' % u_time)
        mi.shader().setUniformData('u_rot_rate_pitch', [phase_rot_pitch_rot_rate])
        mi.shader().setUniformData('u_rot_rate_yaw', [phase_rot_yaw_rot_rate])
        mi.shader().setUniformData('u_start_time_phase', [next_phase_time])
        next_phase = 'harmonic01'
        next_phase_time += 1./phase_rot_pitch_rot_rate/2 + 1./phase_rot_yaw_rot_rate/2 + phase_delay

    elif u_time > next_phase_time and next_phase == 'harmonic01':
        print('Switch to phase HARMONIC01 at time %.1f' % u_time)
        mi.shader().setUniformData('u_use_harmonic01', [True])
        mi.shader().setUniformData('u_start_time_phase', [next_phase_time])
        next_phase = 'harmonic02'
        next_phase_time += phase_harmonic_time + phase_delay

    elif u_time > next_phase_time and next_phase == 'harmonic02':
        print('Switch to phase HARMONIC02 at time %.1f' % u_time)
        mi.shader().setUniformData('u_use_harmonic02', [True])
        mi.shader().setUniformData('u_start_time_phase', [next_phase_time])
        next_phase = None
        next_phase_time += phase_harmonic_time + phase_delay


    # Reset phase params
    elif u_time > next_phase_time - phase_delay:
        mi.shader().setUniformData('u_rot_rate_pitch', [0.])
        mi.shader().setUniformData('u_rot_rate_yaw', [0.])
        mi.shader().setUniformData('u_use_harmonic01', [False])
        mi.shader().setUniformData('u_use_harmonic02', [False])



    mi.shader().setUniformData('u_time', [u_time])
    mi.update()

if __name__ == '__main__':

    frametime = 1./60

    # Start params
    start_delay = 5.
    start_phase = 'transl'
    start_elev = 30
    start_distance = 3
    start_azimuth = 0

    elev_offset = 2.

    # Phase params
    phase_delay = 2.0

    phase_transl_time = 10.

    phase_rot_pitch_rot_rate = 0.05
    phase_rot_yaw_rot_rate = 0.05

    phase_harmonic_time = 10.

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
    window.resize(QtCore.QSize(1000, 900))
    window.show()
    window.setWindowTitle('Stimulus preview')
    window.setCameraPosition(elevation=start_elev+elev_offset, distance=start_distance, azimuth=start_azimuth)

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