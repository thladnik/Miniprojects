from glumpy import app, gl, glm, gloo, transforms
import numpy as np
from scipy.spatial import Delaunay

from Geometry_Helper import SphereHelper, UVSphere

from IPython import embed

vertex_shader = """
    const float pi = 3.14;

    // Transforms SOUTH WEST
    uniform mat4   u_rot_sw; 
    uniform mat4   u_trans_sw;   
    uniform mat4   u_projection_sw;
    uniform float  u_radial_offset_sw;
    uniform float  u_tangent_offset_sw;
    
    // Transforms SOUTH EAST
    uniform mat4   u_rot_se;
    uniform mat4   u_trans_se;
    uniform mat4   u_projection_se;
    uniform float  u_radial_offset_se;
    uniform float  u_tangent_offset_se;
    
    // Transforms NORTH EAST
    uniform mat4   u_rot_ne;
    uniform mat4   u_trans_ne;
    uniform mat4   u_projection_ne;
    uniform float  u_radial_offset_ne;
    uniform float  u_tangent_offset_ne;
    
    // Transforms NORTH WEST
    uniform mat4   u_rot_nw;
    uniform mat4   u_trans_nw;
    uniform mat4   u_projection_nw;
    uniform float  u_radial_offset_nw;
    uniform float  u_tangent_offset_nw;
    
    // Vertex attributes
    attribute vec3 a_cart_pos;      // Cartesian vertex position
    attribute vec2 a_sph_pos;       // Spherical vertex position
    attribute vec2 a_channel;       // Image channel id (1: SW, 2: SE, 3: NE, 4: NW)

    // Variables
    varying vec4 v_cart_pos_transformed;
    varying vec3   v_cart_pos;      // Cartesian vertex position
    varying vec2   v_sph_pos;       // Spherical vertex position
    
    vec4 channelPosition() {
        // SOUTH WEST
        if (a_channel == 1) {
            // First: non-linear projection
            vec4 pos =  u_projection_sw * u_trans_sw * u_rot_sw * vec4(a_cart_pos, 1.0);
            //// Second: linear transformations in image plane (shifting/scaling/rotating 2d image)
            // Radial offset
            pos.xy -= u_radial_offset_sw * pos.w;
            // Rangential offset
            pos.x += u_tangent_offset_sw * pos.w;
            pos.y -= u_tangent_offset_sw * pos.w;
            // Last: return position for vertex
            return pos;
        }
        // SOUTH EAST
        else if (a_channel == 2) {
           // First: non-linear projection
            vec4 pos = u_projection_se * u_trans_se * u_rot_se * vec4(a_cart_pos, 1.0);
            //// Second: linear transformations in image plane (shifting/scaling/rotating 2d image)
            // Radial offset
            pos.x += u_radial_offset_se * pos.w;
            pos.y -= u_radial_offset_se * pos.w;
            // Tangential offset
            pos.xy += u_tangent_offset_se * pos.w;
            // Last: return position for vertex
            return pos;
        }
        // NORTH EAST
        else if (a_channel == 3) {
           // First: non-linear projection
            vec4 pos = u_projection_ne * u_trans_ne * u_rot_ne * vec4(a_cart_pos, 1.0);
            //// Second: linear transformations in image plane (shifting/scaling/rotating 2d image)
            // Radial offset
            pos.xy += u_radial_offset_ne * pos.w;
            // Tangential offset
            pos.x -= u_tangent_offset_ne * pos.w;
            pos.y += u_tangent_offset_ne * pos.w;
            // Last: return position for vertex
            return pos;
        }
        // NORTH WEST
        else if (a_channel == 4) {
           // First: non-linear projection
            vec4 pos = u_projection_nw * u_trans_nw * u_rot_nw * vec4(a_cart_pos, 1.0);
            //// Second: linear transformations in image plane (shifting/scaling/rotating 2d image)
            // Radial offset
            pos.x -= u_radial_offset_nw * pos.w;
            pos.y += u_radial_offset_nw * pos.w;
            // Tangential offset
            pos.xy -= u_tangent_offset_nw * pos.w;
            // Last: return position for vertex
            return pos;
        }
    }
    
    void main()
    {
      v_cart_pos = a_cart_pos;
      v_sph_pos = a_sph_pos;
    
      // Final position
      v_cart_pos_transformed = channelPosition();
      gl_Position = v_cart_pos_transformed;

      <viewport.transform>;
    }
"""

fragment_shader = """
    const float pi = 3.14;
    
    varying vec3 v_cart_pos;
    varying vec2 v_sph_pos;
    
    void main()
    {
        <viewport.clipping>;
    
        // Show checkerboard
        //float c = sin(5.0 * pi * v_cart_pos.x);
        //float c = sin(10.0 * v_sph_pos.x);
        
        // Checkerboard
        float c = sin(10.0 * v_sph_pos.x) * sin(8.0 * v_sph_pos.y);
        
        if (c > 0) {
           c = 1.0;
        } else {
             c = 0.0;
        }
        
        //float c = v_cart_pos_transformed.z;
                
        // Final color
        gl_FragColor = vec4(c, c, c, 1.0);
        
    }
"""


## WINDOW
app.use('qt5')
window = app.Window(width=800, height=800, color=(1, 1, 1, 1))


@window.event
def on_resize(width, height):

    ## Update viewport (center local viewport with aspect = 1)
    if height > width:
        length = width
        x_offset = 0
        y_offset = (height - length) // 2
    else:
        length = height
        x_offset = (width - length) // 2
        y_offset = 0
    program['viewport']['global'] = (0, 0, width, height)
    program['viewport']['local'] = (x_offset, y_offset, length, length)

    ## Set default image channel parameters
    std_trans_distance = -10.
    std_fov = 30.
    std_elevation_rot = 90.
    std_radial_offset = 0.5
    std_tangent_offset = 0.2

    elevation_rot_sw = 0.
    elevation_rot_se = 0.
    elevation_rot_ne = 0.
    elevation_rot_nw = 0.

    azimuth_rot_sw = 0.
    azimuth_rot_se = 0.
    azimuth_rot_ne = 0.
    azimuth_rot_nw = 0.

    ## SOUTH WEST
    # Non-linear transformations
    rot_axis_sw = (-1, 1, 0)
    u_projection = glm.perspective(std_fov, 1.0, 0.01, 1000.0)
    u_rot = np.eye(4, dtype=np.float32)
    glm.rotate(u_rot, azimuth_rot_sw, 0, 0, 1)  # Rotate around equator
    glm.rotate(u_rot, std_elevation_rot-elevation_rot_sw, *rot_axis_sw)  # Rotate around current azim. major circle
    u_trans = glm.translation(0., 0., std_trans_distance)
    program['u_trans_sw'] = u_trans
    program['u_rot_sw'] = u_rot
    program['u_projection_sw'] = u_projection
    # Linear image plane transformations
    program['u_radial_offset_sw'] = std_radial_offset
    program['u_tangent_offset_sw'] = std_tangent_offset

    ## SOUTH EAST
    # Non-linear transformations
    rot_axis_se = (-1, -1, 0)
    u_projection = glm.perspective(std_fov, 1.0, 0.01, 1000.0)
    u_rot = np.eye(4, dtype=np.float32)
    glm.rotate(u_rot, azimuth_rot_se, 0, 0, 1)  # Rotate around equator
    glm.rotate(u_rot, std_elevation_rot-elevation_rot_se, *rot_axis_se)  # Rotate around current azim. major circle
    u_trans = glm.translation(0., 0., std_trans_distance)
    program['u_trans_se'] = u_trans
    program['u_rot_se'] = u_rot
    program['u_projection_se'] = u_projection
    # Linear image plane transformations
    program['u_radial_offset_se'] = std_radial_offset
    program['u_tangent_offset_se'] = std_tangent_offset

    rot_axis_ne = (1, -1, 0)
    u_projection = glm.perspective(std_fov, 1.0, 0.01, 1000.0)
    u_rot = np.eye(4, dtype=np.float32)
    glm.rotate(u_rot, azimuth_rot_ne, 0, 0, 1)  # Rotate around equator
    glm.rotate(u_rot, std_elevation_rot-elevation_rot_ne, *rot_axis_ne)  # Rotate around current azim. major circle
    u_trans = glm.translation(0., 0., std_trans_distance)
    program['u_trans_ne'] = u_trans
    program['u_rot_ne'] = u_rot
    program['u_projection_ne'] = u_projection
    # Linear image plane transformations
    program['u_radial_offset_ne'] = std_radial_offset
    program['u_tangent_offset_ne'] = std_tangent_offset

    rot_axis_nw = (1, 1, 0)
    u_projection = glm.perspective(std_fov, 1.0, 0.01, 1000.0)
    u_rot = np.eye(4, dtype=np.float32)
    glm.rotate(u_rot, azimuth_rot_nw, 0, 0, 1)  # Rotate around equator
    glm.rotate(u_rot, std_elevation_rot-elevation_rot_nw, *rot_axis_nw)  # Rotate around current azim. major circle
    u_trans = glm.translation(0., 0., std_trans_distance)
    program['u_trans_nw'] = u_trans
    program['u_rot_nw'] = u_rot
    program['u_projection_nw'] = u_projection
    # Linear image plane transformations
    program['u_radial_offset_nw'] = std_radial_offset
    program['u_tangent_offset_nw'] = std_tangent_offset

@window.event
def on_draw(dt):
    #print(dt)
    window.clear(color=(0.0, 0.0, 0.0, 1.0))  # black
    #gl.glDisable(gl.GL_BLEND)
    gl.glEnable(gl.GL_DEPTH_TEST)

    program.draw(gl.GL_TRIANGLES, indices)
    #program.draw(gl.GL_POINTS, vertices)

sphere = UVSphere(80, 40, upper_phi=np.pi/4)
all_verts = sphere.getVertices()
all_faces = sphere.getFaceIndices()
all_sph_pos = sphere.getSphericalCoords()

### BUILD SEGMENTED SPHERE

orientations = ['sw', 'se', 'ne', 'nw']
verts = dict()
faces = dict()
sph_pos = dict()
channel = dict()
for i, orient in enumerate(orientations):
    theta_center = -3*np.pi/4 + i * np.pi/2
    vert_mask = SphereHelper.getAzElLimitedMask(theta_center-np.pi/4, theta_center+np.pi/4,
                                                -np.inf, np.inf, all_verts)

    verts[orient] = all_verts[vert_mask]
    sph_pos[orient] = all_sph_pos[vert_mask]
    channel[orient] = (i+1) * np.ones((verts[orient].shape[0], 2))
    faces[orient] = Delaunay(verts[orient]).convex_hull

## CREATE BUFFERS
v = np.concatenate([verts[orient] for orient in orientations], axis=0)
# Vertex buffer
vertices = np.zeros(v.shape[0],
                    [('a_cart_pos', np.float32, 3),
                     ('a_sph_pos', np.float32, 2),
                     ('a_channel', np.float32, 2)])
vertices['a_cart_pos'] = v.astype(np.float32)
vertices['a_sph_pos'] = np.concatenate([sph_pos[orient] for orient in orientations], axis=0).astype(np.float32)
vertices['a_channel'] = np.concatenate([channel[orient] for orient in orientations], axis=0).astype(np.float32)
vertices = vertices.view(gloo.VertexBuffer)
# Index buffer
indices = np.zeros((0, 3))
startidx = 0
for orient in orientations:
    indices = np.concatenate([indices, startidx+faces[orient]], axis=0)
    startidx += verts[orient].shape[0]
indices = indices.astype(np.uint32).view(gloo.IndexBuffer)

## CREATE PROGRAM
program = gloo.Program(vertex_shader, fragment_shader)
program.bind(vertices)

## Set and attach viewport
program['viewport'] = transforms.Viewport()
window.attach(program['viewport'])

## RUN
app.run(framerate=30)