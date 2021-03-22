import {defs, tiny} from './examples/common.js';
import { Shape_From_File } from './examples/obj-file-demo.js';

const {
    Vector, Vector3, vec, vec3, vec4, color, hex_color, Shader, Matrix, Mat4, Light, Shape, Texture, Material, Scene, camera_transform
} = tiny;

const {Cube, Axis_Arrows, Textured_Phong, Fake_Bump_Map} = defs

export class Project extends Scene {
    constructor() {
        // constructor(): Scenes begin by populating initial values like the Shapes and Materials they'll need.
        super();

        // At the beginning of our program, load one of each of these shape definitions onto the GPU.
        this.shapes = {
            box: new defs.Bump_Box(),
            sky_sphere: new defs.Subdivision_Sphere(4),
            tree_stump:  new Shape_From_File("assets/naked_tree.obj"),
            plane: new defs.Square(),
            circle: new defs.Regular_2D_Polygon(1, 15),
            new_tree: new Shape_From_File("assets/new_tree.obj"),
            final_tree: new Shape_From_File("assets/final_tree.obj"),
            leaves: new Shape_From_File("assets/leaves.obj"),
            leaves2: new Shape_From_File("assets/leaves2.obj"),
            final_leaves: new Shape_From_File("assets/final_tree_leaves.obj"),
            lanturn: new Shape_From_File("assets/lanturn.obj"),
            lanturn_handle: new Shape_From_File("assets/lanturn_handle.obj"),
            // TODO:  Fill in as many additional shape instances as needed in this key/value table.
            //        (Requirement 1)
        };

        // *** Materials
        this.materials = {
            my_bump: new Material(new Bump_Phong(1), {
                color: hex_color("#000000"),
                ambient: .5, diffusivity: 0.5, specularity: 0.3,
                texture: new Texture("assets/grass.jpg", "LINEAR_MIPMAP_LINEAR"),
                bump_map_x: new Texture("assets/grass_grad_x.jpg"),
                bump_map_y: new Texture("assets/grass_grad_y.jpg"),
            }),
            fake_bump: new Material(new defs.Fake_Bump_Map(1), {
                color: hex_color("#000000"),
                ambient: .5, diffusivity: 0.5, specularity: 0.3,
                texture: new Texture("assets/grass.jpg", "LINEAR_MIPMAP_LINEAR")
            }),
            non_bump: new Material(new Textured_Phong(1), {
                color: hex_color("#000000"),
                ambient: .5, diffusivity: 0.5, specularity: 0.3,
                texture: new Texture("assets/grass.jpg")
            }),
            test: new Material(new Fake_Bump_Map(1), {
                color: hex_color("#000000"),
                ambient: .5, diffusivity: 0.5, specularity: 0.4,
                texture: new Texture("assets/rock.png")
            }),
            test2: new Material(new Gouraud_Shader(),
                {ambient: .4, diffusivity: .6, color: hex_color("#992828")}),
            ring: new Material(new Ring_Shader()),
            texture_sample: new Material(new Textured_Phong(), {
                color: hex_color("#999999"),
                ambient: .5, diffusivity: 0.1, specularity: 0.1,
                texture: new Texture("assets/sky.jpg", "LINEAR_MIPMAP_LINEAR")
            }),
            grass: new Material(new Fake_Bump_Map(1), {
                color: hex_color("#000000"),
                ambient: .5, diffusivity: 0.5, specularity: 0.4,
                texture: new Texture("assets/grass.jpg")
            }),
            tree_trunk: new Material(new defs.Phong_Shader(1), {color: hex_color("#d2691e"), ambient: .05, diffusivity: 0.9, specularity: 0.2}),
            leaves_text: new Material(new defs.Phong_Shader(1), {color: hex_color("#336600"), ambient: .05, diffusivity: 0.9, specularity: 0.2}),
               
        }
        this.shapes.plane.arrays.texture_coord = this.shapes.plane.arrays.texture_coord.map(function(x) {return x.times(15)});
        
        //height of player
        this.height = 1;


        // chunk vars
        this.chunk_size = 10;
        this.num_chunks_x = 3;
        this.num_chunks_y = 3;

        // Skybox vars
        this.sky_shape_x = this.chunk_size * this.num_chunks_x;
        this.sky_shape_y = this.chunk_size * this.num_chunks_y;
        this.sky_shape_z = 100;
        
        // Biome vars
        this.biome_memory = [];
        // Eugene: For each of the 3 biome types below in biome_textures,
        // please edit the material to be unique of the tree and leaf 
        // materials. If you have time, please also edit the material
        // for the ground of each biome. Feel free to import additional
        // textures into the project (like instead of grass, sand, etc).
        this.biome_textures = {
            grassland : {
                ground : new Material(new Textured_Phong(1), {
                    color: hex_color("#000000"),
                    ambient: .5, diffusivity: 0.5, specularity: 0.4,
                    texture: new Texture("assets/grass.jpg", "LINEAR_MIPMAP_LINEAR")
                    }),
                sky : new Material(new Textured_Phong(1), {
                    color: hex_color("#999999"),
                    ambient: .5, diffusivity: 0.1, specularity: 0.1,
                    texture: new Texture("assets/sky.jpg", "LINEAR_MIPMAP_LINEAR")
                    }),                
                tree : new Material(new Fake_Bump_Map(1), {
                    color: hex_color("#d2691e"), 
                    ambient: .05, 
                    diffusivity: 0.9, 
                    specularity: 0.2
                    }),
                leaf : new Material(new Textured_Phong(1), { // Eugene: like here on this line
                    color: hex_color("#336600"),
                    ambient: .05, diffusivity: 0.9, specularity: 0.2
                    }),
                totem : new Material(new Textured_Phong(1), {
                    color: hex_color("#999999"),
                    ambient: .5, diffusivity: 0.1, specularity: 0.1,
                    texture: new Texture("assets/sky.jpg", "LINEAR_MIPMAP_LINEAR")
                    }),
            },
            desert : {
                ground : new Material(new Fake_Bump_Map(1), {
                    color: hex_color("#000000"),
                    ambient: .5, diffusivity: 0.5, specularity: 0.4,
                    texture: new Texture("assets/sand.jpg")
                    }),
                sky : new Material(new Textured_Phong(1), {
                    color: hex_color("#0000b0"),
                    ambient: .5, diffusivity: 0.1, specularity: 0.1,
                    texture: new Texture("assets/desert_sky.jpg", "LINEAR_MIPMAP_LINEAR")
                    }),                
                tree : new Material(new Fake_Bump_Map(1), { // Eugene: like here on this line
                    color: hex_color("#580000"),
                    ambient: .5, 
                    diffusivity: 0.5, 
                    specularity: 0.1,
                    }),
                leaf : new Material(new Textured_Phong(1), { // Eugene: like here on this line
                    color: hex_color("#ff32b1"),
                    ambient: .05, diffusivity: 0.9, specularity: 0.2
                    }),
                totem : new Material(new Textured_Phong(1), {
                    color: hex_color("#999999"),
                    ambient: .5, diffusivity: 0.1, specularity: 0.1,
                    texture: new Texture("assets/sky.jpg", "LINEAR_MIPMAP_LINEAR")
                    }),
            },
            badland : {
                ground : new Material(new Fake_Bump_Map(1), {
                    color: hex_color("#000000"),
                    ambient: .5, diffusivity: 0.5, specularity: 0.4,
                    texture: new Texture("assets/badlands.jpg")
                    }),
                sky : new Material(new Fake_Bump_Map(1), {
                    color: hex_color("#000000"),
                    ambient: .5, diffusivity: 0.1, specularity: 0.1,
                    texture: new Texture("assets/badlands_sky.jpg")
                    }),                
                tree : new Material(new Fake_Bump_Map(1), { // Eugene: like here on this line
                    color: hex_color("#d2b48c"),
                    ambient: .5, diffusivity: 0.4, specularity: 0.4,
                    }),
                leaf : new Material(new Fake_Bump_Map(1), { // Eugene: like here on this line
                    color: hex_color("#33ccff"),
                    ambient: .5, diffusivity: 0.1, specularity: 0.1,
                    }),
                totem : new Material(new Textured_Phong(1), {
                    color: hex_color("#999999"),
                    ambient: .5, diffusivity: 0.1, specularity: 0.1,
                    texture: new Texture("assets/sky.jpg", "LINEAR_MIPMAP_LINEAR")
                    }),
            },
        };
        this.biome_names = [this.biome_textures.grassland,
                            this.biome_textures.desert,
                            this.biome_textures.badland,];
        this.num_biomes = this.biome_names.length;
        this.biome_trees = [
            {
                tree : this.shapes.tree_stump,
                tree_offset : vec3(0, 1.7, 0),
                leaf : this.shapes.leaves2,
                leaf_scale : vec3(1.1, 1.1, 1.1),
                leaf_offset : vec3(0.1, 0.6, 0),
            },
            {
                tree : this.shapes.new_tree,
                tree_offset : vec3(0, 1.7, 0),
                leaf : this.shapes.leaves,
                leaf_scale : vec3(1.5, 1.6, 1.5),
                leaf_offset : vec3(0, 0.4, 0),
            },
            {
                tree: this.shapes.final_tree,
                tree_offset : vec3(0, 2, 0),
                leaf : this.shapes.final_leaves,
                leaf_scale : vec3(1.7, 1.7, 1.7),
                leaf_offset : vec3(-.2, .9, 0),  
            },
        ];
        this.num_trees = 3;
        this.tree_num = -1;
        this.tree_combo = -1;
        // Camera vars
        this.initial_camera_location = Mat4.look_at(vec3(0, this.height, 0), vec3(0, this.height, -3), vec3(0, 1, 0));

    }

    make_control_panel() {
        // Draw the scene's buttons, setup their actions and keyboard shortcuts, and monitor live measurements.
    }

    euclidean_dist(x2, x1, z2, z1) {
        let dist = 0;
        dist = Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(z2 - z1, 2));
        return dist;
    }

    //pass in either model_transform_chunk or temp_transform
    populate(player_transform, context, program_state, biome_num, chunk_size){
        
    }

    display(context, program_state) {
        // display():  Called once per frame of animation.
        // Setup -- This part sets up the scene's overall camera matrix, projection matrix, and lights:
        if (!context.scratchpad.controls) {
            this.children.push(context.scratchpad.controls = new defs.Movement_Controls());
            // Define the global camera and projection matrices, which are stored in program_state.
            program_state.set_camera(this.initial_camera_location);
        }
        program_state.projection_transform = Mat4.perspective(
            Math.PI / 4, context.width / context.height, .1, 1000);

        // TODO: Adjust lighting
        const light_position = vec4(program_state.camera_transform[0][3], this.height, program_state.camera_transform[2][3], 1);
        program_state.lights = [new Light(light_position, color(1, 1, 1, 1), 1000)];
        //program_state.lights = [];

        const t = program_state.animation_time / 1000, dt = program_state.animation_delta_time / 1000;


//======Biomes TODO: everything
        let character_position = vec3(program_state.camera_transform[0][3], this.height, program_state.camera_transform[2][3]);
        /* equivalent to the player's distance from the origin (measured in chunks).
         * This can be thought of as a "ring" around the center that all chunks 
         * in that ring share. This var is used to calculate unique textures/ 
         * obj spawn positions in each chunk. */
        let curr_biome = Math.floor(character_position.norm() / this.chunk_size); // TODO: test 4 correctness.
        
        /* If we haven't visited this biome "ring" before: */
        if ( !(curr_biome in this.biome_memory) ) {
            // used to choose textures randomly
            var chosen_biome = Math.floor(Math.random() * (this.num_biomes));
            
            // save the chosen texture into the texture_memory dict
            this.biome_memory.push({num : curr_biome, type : this.biome_names[chosen_biome]});
            //num_biomes = Object.keys(this.biome_memory).length;

            this.tree_num = Math.floor(Math.random() * (this.num_trees));
            this.tree_combo = this.biome_trees[this.tree_num];
        }



//======Ground Chunks TODO: rename section?
        /* Vector representing the distance to the origin, as measured in 
         * chunks. */
        let chunks_2_org = vec3(Math.floor(character_position[0] / this.chunk_size),
                                            Math.floor(character_position[1] / this.chunk_size),
                                            Math.floor(character_position[2] / this.chunk_size));
        /* Useful for determining the position of the player with respect to
         * the current chunk */
        let chunk_remainder_vec = vec3(character_position[0] % this.chunk_size,
                                            character_position[1] % this.chunk_size,
                                            character_position[2] % this.chunk_size);
        //console.log(Math.floor(chunks_2_org[2]));
        /* Stores the matrix that encodes where to place the chunk beneath
         * the player. We will perform operation on this matrix to decide
         * where to place neighboring chunks in the loop below. */
        let model_transform_chunk = Mat4.identity();
        model_transform_chunk = model_transform_chunk.times(Mat4.rotation( (Math.PI / 2), 1, 0, 0 ));
        model_transform_chunk = model_transform_chunk.times(Mat4.scale(this.chunk_size, this.chunk_size, 0));        
        model_transform_chunk = model_transform_chunk
            .times(Mat4.translation(Math.floor(chunks_2_org[0]), Math.floor(chunks_2_org[2]), 0));
                            // We are translating on the xy-plane, because the ground
                            // chunks were already rotated once across the
                            // x-axis. Due to this rotation, the above statement
                            // actually translates with respect to the xz-plane.
        /* This for loop iteratively places num_chunks_x by num_chunks_y squared_scale
         * under the player's current x,x-position. */
        var i;
        var j;
        let offset_x = -1 * Math.floor(this.num_chunks_x / 2);
        let offset_y = -1 * Math.floor(this.num_chunks_y / 2);
        
        for (i = 0; i < this.num_chunks_x; i++) {
            for (j = 0; j < this.num_chunks_y; j++) {
                // 0.5 is added to translation, otherwise chunks were left-biased.
                let temp_transform = model_transform_chunk
                            .times(Mat4.translation(offset_x + i + 0.5, offset_y + j + 0.5, 0));
                
                this.draw_biome(context, program_state, temp_transform, curr_biome);
                /* we want to change the curr_biome to reflect the chunk's 
                 * i & j values before calling draw_biome(). If we don't, 
                 * all 9 chunks will change to the new biome type at the 
                 * same time, which is gross. */
                 // TODO: this isn't right.
//                  if ( curr_biome > this.num_chunks_x ) {
//                      curr_biome = curr_biome + offset_x + i;
//                  }
                // TODO: customize ground plane given chosen biome.
            }            
        }


        
//======Skybox TODO: customize skybox given chosen biome.
        let model_transform_sky = Mat4.identity();
        model_transform_sky = model_transform_sky.times(Mat4.scale(this.sky_shape_x, this.sky_shape_y, this.sky_shape_z));
        model_transform_sky = model_transform_sky
            .times(Mat4.translation(character_position[0] / this.sky_shape_x, 0, character_position[2] / this.sky_shape_z));
        this.shapes.sky_sphere.draw(context, program_state, model_transform_sky, this.biome_memory[curr_biome].type.sky);
        
//======Monument TODO: customize skybox given chosen biome.
        let model_transform_object = Mat4.identity();
        model_transform_object = model_transform_object.times(Mat4.translation(0, 1.0, 10));
        this.shapes.box.draw(context, program_state, model_transform_object, this.materials.test);
        
        let model_transform_leaves2 = Mat4.identity();
        model_transform_leaves2 = model_transform_leaves2.times(Mat4.translation(0, 3, 10));
        this.shapes.box.draw(context, program_state, model_transform_leaves2, this.materials.test);
        
        let dist_val = character_position.norm();
        let sine = 0.1 * Math.sin(dist_val);
        let model_transform_lanturn = Mat4.identity()
                                    .times(program_state.camera_transform)
                                    .times(Mat4.translation(.5, -.8 + sine, -2))
                                    .times(Mat4.scale(.5, .5, .5));
                                    //.times(Mat4.translation(0, -0,75, 4));
        this.shapes.lanturn.draw(context, program_state, model_transform_lanturn, this.materials.leaves_text);                        

//         model_transform_leaves2 = Mat4.identity();
//         model_transform_leaves2 = model_transform_leaves2.times(Mat4.translation(7.1, 2.3, -15))
//                                                          .times(Mat4.scale(1.1, 1.1, 1.1));
//         this.shapes.leaves2.draw(context, program_state, model_transform_leaves2, this.materials.leaves_text);
//         let model_transform_new_tree = Mat4.identity();
//         model_transform_new_tree = model_transform_new_tree.times(Mat4.translation(-5, 1.7, -10));
//         this.shapes.new_tree.draw(context, program_state, model_transform_new_tree, this.materials.tree_trunk.override({color: hex_color("#1b0000")}));
//         let model_transform_new_leaves = Mat4.identity();
//         model_transform_new_leaves = model_transform_new_leaves.times(Mat4.translation(-5, 2.1, -10))
//                                                                .times(Mat4.scale(1.5, 1.6, 1.5));
//         this.shapes.leaves.draw(context, program_state, model_transform_new_leaves, this.materials.leaves_text);

        
        // TODO: fix character.
//         let model_transform_box = Mat4.identity();
//         model_transform_box = model_transform_box.times(Mat4.translation(-2, 1, -4));                                                       
//         this.shapes.box.draw(context, program_state, model_transform_box, this.materials.fake_bump);
//         model_transform_box = model_transform_box.times(Mat4.translation(2, 0, 0)); 
//         this.shapes.box.draw(context, program_state, model_transform_box, this.materials.my_bump);
//         model_transform_box = model_transform_box.times(Mat4.translation(2, 0, 0)); 
//         this.shapes.box.draw(context, program_state, model_transform_box, this.materials.non_bump);
    } // end display()
    
    // TODO: include offset so that all 9 chunks dont change at once.
    draw_biome(context, program_state, chunk_transform, biome) {
        // Eugene: this gets passed into draw_tree. Use this to 
        // choose bark and leaf texture.
        var chosen_biome = this.biome_memory[biome].type;
        let chunk_center = chunk_transform.times( vec4( 0,0,0,1 ) ).to3();

        // choose tree/ leaf combo
        // Eugene: Right now we choose which type of tree to draw at random
        // every frame. We need to somehow store which type of tree was chosen.
        
        // TODO: Decide to draw tree
        var draw_it = false;
        var i = 0;
        this.draw_tree(context, program_state, this.tree_combo, chunk_center, chosen_biome);
        this.shapes.plane.draw(context, program_state, chunk_transform, chosen_biome.ground);
    }// end draw_biome

    draw_tree(context, program_state, tree_choice, center_vec, chosen_biome) {
        // Draw tree
        var tree_transform = Mat4.identity().times(Mat4.translation(center_vec[0], center_vec[1], center_vec[2]));
        tree_transform = tree_transform
            .times(Mat4.translation(tree_choice.tree_offset[0], tree_choice.tree_offset[1], tree_choice.tree_offset[2]));
        tree_choice.tree.draw(context, program_state, tree_transform, chosen_biome.tree);
        
        // Draw leaves
        let model_transform_leaves2 = tree_transform
            .times(Mat4.translation(tree_choice.leaf_offset[0], tree_choice.leaf_offset[1], tree_choice.leaf_offset[2]))
            .times(Mat4.scale(tree_choice.leaf_scale[0], tree_choice.leaf_scale[1], tree_choice.leaf_scale[2]));        
        tree_choice.leaf.draw(context, program_state, model_transform_leaves2, chosen_biome.leaf);
    }
} // end project class

class Bump_Phong extends Textured_Phong
{ 

  shared_glsl_code()           // ********* SHARED CODE, INCLUDED IN BOTH SHADERS *********
    { return ` precision mediump float;
        const int N_LIGHTS = ` + this.num_lights + `;
        uniform float ambient, diffusivity, specularity, smoothness;
        uniform vec4 light_positions_or_vectors[N_LIGHTS], light_colors[N_LIGHTS];
        uniform float light_attenuation_factors[N_LIGHTS];
        uniform vec4 shape_color;
        uniform vec3 squared_scale, camera_center;

                              // Specifier "varying" means a variable's final value will be passed from the vertex shader
                              // on to the next phase (fragment shader), then interpolated per-fragment, weighted by the
                              // pixel fragment's proximity to each of the 3 vertices (barycentric interpolation).
        varying vec3 N, T, vertex_worldspace;
                                             // ***** PHONG SHADING HAPPENS HERE: *****                                       
        vec3 phong_model_lights( vec3 N, vec3 vertex_worldspace )
          {                                        // phong_model_lights():  Add up the lights' contributions.
            vec3 E = normalize( camera_center - vertex_worldspace );
            vec3 result = vec3( 0.0 );
            for(int i = 0; i < N_LIGHTS; i++)
              {
                            // Lights store homogeneous coords - either a position or vector.  If w is 0, the 
                            // light will appear directional (uniform direction from all points), and we 
                            // simply obtain a vector towards the light by directly using the stored value.
                            // Otherwise if w is 1 it will appear as a point light -- compute the vector to 
                            // the point light's location from the current surface point.  In either case, 
                            // fade (attenuate) the light as the vector needed to reach it gets longer.  
                vec3 surface_to_light_vector = light_positions_or_vectors[i].xyz - 
                                               light_positions_or_vectors[i].w * vertex_worldspace;                                             
                float distance_to_light = length( surface_to_light_vector );

                vec3 L = normalize( surface_to_light_vector );
                vec3 H = normalize( L + E );
                                                  // Compute the diffuse and specular components from the Phong
                                                  // Reflection Model, using Blinn's "halfway vector" method:
                float diffuse  =      max( dot( N, L ), 0.0 );
                float specular = pow( max( dot( N, H ), 0.0 ), smoothness );
                float attenuation = 1.0 / (1.0 + light_attenuation_factors[i] * distance_to_light * distance_to_light );
                
                
                vec3 light_contribution = shape_color.xyz * light_colors[i].xyz * diffusivity * diffuse
                                                          + light_colors[i].xyz * specularity * specular;

                result += attenuation * light_contribution;
              }
            return result;
          } ` ;
    }
  vertex_glsl_code()           // ********* VERTEX SHADER *********
    { return this.shared_glsl_code() + `
        varying vec2 f_tex_coord;
        attribute vec3 position, normal, tangents;                            // Position is expressed in object coordinates.
        attribute vec2 texture_coord;
        
        uniform mat4 model_transform;
        uniform mat4 projection_camera_model_transform;
        

        void main()
          {                                                                   // The vertex's final resting place (in NDCS):
            gl_Position = projection_camera_model_transform * vec4( position, 1.0 );
            
            T = normalize( vec3( model_transform * vec4( tangents, 0.0 ) ) );
                                                                              // The final normal vector in screen space.
            N = normalize( mat3( model_transform ) * normal / squared_scale);
            
            vertex_worldspace = ( model_transform * vec4( position, 1.0 ) ).xyz;
                                              // Turn the per-vertex texture coordinate into an interpolated variable.
            f_tex_coord = texture_coord;
          } ` ;
    }
  fragment_glsl_code()
    {                            // ********* FRAGMENT SHADER ********* 
      return this.shared_glsl_code() + `
        varying vec2 f_tex_coord;
        uniform sampler2D texture;
        uniform sampler2D bump_map_x;
        uniform sampler2D bump_map_y;


        void main()
          {                                                          // Sample the texture image in the correct place:

            vec3 my_normal = normalize( N );
            vec3 my_tan = normalize( T );
            vec3 my_binorm = cross(my_normal, my_tan);
            vec4 tex_color = texture2D( texture, f_tex_coord );
            if( tex_color.w < .01 ) discard;
                             // Slightly disturb normals based on sampling the same image that was used for texturing:
            float my_grad_x = texture2D(bump_map_x, f_tex_coord).r;
            float my_grad_y = texture2D(bump_map_y, f_tex_coord).r;
            vec3 bumped_N  = my_tan * my_grad_x + my_binorm * my_grad_y;
                                                                     // Compute an initial (ambient) color:
            gl_FragColor = vec4( ( tex_color.xyz + shape_color.xyz ) * ambient, shape_color.w * tex_color.w ); 
                                                                     // Compute the final color with contributions from lights:
            my_normal += bumped_N;
            gl_FragColor.xyz += phong_model_lights( normalize( my_normal ), vertex_worldspace );
          } ` ;
    }

}

class Gouraud_Shader extends Shader {
    // This is a Shader using Phong_Shader as template
    // TODO: Modify the glsl coder here to create a Gouraud Shader (Planet 2)

    constructor(num_lights = 2) {
        super();
        this.num_lights = num_lights;
    }

    shared_glsl_code() {
        // ********* SHARED CODE, INCLUDED IN BOTH SHADERS *********
        return ` 
        precision mediump float;
        const int N_LIGHTS = ` + this.num_lights + `;
        uniform float ambient, diffusivity, specularity, smoothness;
        uniform vec4 light_positions_or_vectors[N_LIGHTS], light_colors[N_LIGHTS];
        uniform float light_attenuation_factors[N_LIGHTS];
        uniform vec4 shape_color;
        uniform vec3 squared_scale, camera_center;

        // Specifier "varying" means a variable's final value will be passed from the vertex shader
        // on to the next phase (fragment shader), then interpolated per-fragment, weighted by the
        // pixel fragment's proximity to each of the 3 vertices (barycentric interpolation).
        varying vec3 N, vertex_worldspace;
        varying vec4 COLOR;
        // ***** PHONG SHADING HAPPENS HERE: *****                                       
        vec3 phong_model_lights( vec3 N, vec3 vertex_worldspace ){                                        
            // phong_model_lights():  Add up the lights' contributions.
            vec3 E = normalize( camera_center - vertex_worldspace );
            vec3 result = vec3( 0.0 );
            for(int i = 0; i < N_LIGHTS; i++){
                // Lights store homogeneous coords - either a position or vector.  If w is 0, the 
                // light will appear directional (uniform direction from all points), and we 
                // simply obtain a vector towards the light by directly using the stored value.
                // Otherwise if w is 1 it will appear as a point light -- compute the vector to 
                // the point light's location from the current surface point.  In either case, 
                // fade (attenuate) the light as the vector needed to reach it gets longer.  
                vec3 surface_to_light_vector = light_positions_or_vectors[i].xyz - 
                                               light_positions_or_vectors[i].w * vertex_worldspace;                                             
                float distance_to_light = length( surface_to_light_vector );

                vec3 L = normalize( surface_to_light_vector );
                vec3 H = normalize( L + E );
                // Compute the diffuse and specular components from the Phong
                // Reflection Model, using Blinn's "halfway vector" method:
                float diffuse  =      max( dot( N, L ), 0.0 );
                float specular = pow( max( dot( N, H ), 0.0 ), smoothness );
                float attenuation = 1.0 / (1.0 + light_attenuation_factors[i] * distance_to_light * distance_to_light );
                
                vec3 light_contribution = shape_color.xyz * light_colors[i].xyz * diffusivity * diffuse
                                                          + light_colors[i].xyz * specularity * specular;
                result += attenuation * light_contribution;
            }
            return result;
        } `;
    }

    vertex_glsl_code() {
        // ********* VERTEX SHADER *********
        return this.shared_glsl_code() + `
            attribute vec3 position, normal;                            
            // Position is expressed in object coordinates.
            
            uniform mat4 model_transform;
            uniform mat4 projection_camera_model_transform;
    
            void main(){                                                                   
                // The vertex's final resting place (in NDCS):
                gl_Position = projection_camera_model_transform * vec4( position, 1.0 );
                // The final normal vector in screen space.
                N = normalize( mat3( model_transform ) * normal / squared_scale);
                vertex_worldspace = ( model_transform * vec4( position, 1.0 ) ).xyz;

                COLOR = vec4( shape_color.xyz * ambient, shape_color.w );
                COLOR.xyz += phong_model_lights( normalize( N ), vertex_worldspace );
                //COLOR.xyz += phong_model_lights( shape_color.xyz, vertex_worldspace );
            } `;
    }

    fragment_glsl_code() {
        // ********* FRAGMENT SHADER *********
        // A fragment is a pixel that's overlapped by the current triangle.
        // Fragments affect the final image or get discarded due to depth.
        return this.shared_glsl_code() + `
            void main(){
                gl_FragColor = COLOR;

                                                                           
                // Compute an initial (ambient) color:
                //gl_FragColor = vec4( shape_color.xyz * ambient, shape_color.w );
                // Compute the final color with contributions from lights:
                //gl_FragColor.xyz += phong_model_lights( normalize( N ), vertex_worldspace );
            } `;
    }

    send_material(gl, gpu, material) {
        // send_material(): Send the desired shape-wide material qualities to the
        // graphics card, where they will tweak the Phong lighting formula.
        gl.uniform4fv(gpu.shape_color, material.color);
        gl.uniform1f(gpu.ambient, material.ambient);
        gl.uniform1f(gpu.diffusivity, material.diffusivity);
        gl.uniform1f(gpu.specularity, material.specularity);
        gl.uniform1f(gpu.smoothness, material.smoothness);
    }

    send_gpu_state(gl, gpu, gpu_state, model_transform) {
        // send_gpu_state():  Send the state of our whole drawing context to the GPU.
        const O = vec4(0, 0, 0, 1), camera_center = gpu_state.camera_transform.times(O).to3();
        gl.uniform3fv(gpu.camera_center, camera_center);
        // Use the squared scale trick from "Eric's blog" instead of inverse transpose matrix:
        const squared_scale = model_transform.reduce(
            (acc, r) => {
                return acc.plus(vec4(...r).times_pairwise(r))
            }, vec4(0, 0, 0, 0)).to3();
        gl.uniform3fv(gpu.squared_scale, squared_scale);
        // Send the current matrices to the shader.  Go ahead and pre-compute
        // the products we'll need of the of the three special matrices and just
        // cache and send those.  They will be the same throughout this draw
        // call, and thus across each instance of the vertex shader.
        // Transpose them since the GPU expects matrices as column-major arrays.
        const PCM = gpu_state.projection_transform.times(gpu_state.camera_inverse).times(model_transform);
        gl.uniformMatrix4fv(gpu.model_transform, false, Matrix.flatten_2D_to_1D(model_transform.transposed()));
        gl.uniformMatrix4fv(gpu.projection_camera_model_transform, false, Matrix.flatten_2D_to_1D(PCM.transposed()));

        // Omitting lights will show only the material color, scaled by the ambient term:
        if (!gpu_state.lights.length)
            return;

        const light_positions_flattened = [], light_colors_flattened = [];
        for (let i = 0; i < 4 * gpu_state.lights.length; i++) {
            light_positions_flattened.push(gpu_state.lights[Math.floor(i / 4)].position[i % 4]);
            light_colors_flattened.push(gpu_state.lights[Math.floor(i / 4)].color[i % 4]);
        }
        gl.uniform4fv(gpu.light_positions_or_vectors, light_positions_flattened);
        gl.uniform4fv(gpu.light_colors, light_colors_flattened);
        gl.uniform1fv(gpu.light_attenuation_factors, gpu_state.lights.map(l => l.attenuation));
    }

    update_GPU(context, gpu_addresses, gpu_state, model_transform, material) {
        // update_GPU(): Define how to synchronize our JavaScript's variables to the GPU's.  This is where the shader
        // recieves ALL of its inputs.  Every value the GPU wants is divided into two categories:  Values that belong
        // to individual objects being drawn (which we call "Material") and values belonging to the whole scene or
        // program (which we call the "Program_State").  Send both a material and a program state to the shaders
        // within this function, one data field at a time, to fully initialize the shader for a draw.

        // Fill in any missing fields in the Material object with custom defaults for this shader:
        const defaults = {color: color(0, 0, 0, 1), ambient: 0, diffusivity: 1, specularity: 1, smoothness: 40};
        material = Object.assign({}, defaults, material);

        this.send_material(context, gpu_addresses, material);
        this.send_gpu_state(context, gpu_addresses, gpu_state, model_transform);
    }
}

class Ring_Shader extends Shader {
    update_GPU(context, gpu_addresses, graphics_state, model_transform, material) {
        // update_GPU():  Defining how to synchronize our JavaScript's variables to the GPU's:
        const [P, C, M] = [graphics_state.projection_transform, graphics_state.camera_inverse, model_transform],
            PCM = P.times(C).times(M);
        context.uniformMatrix4fv(gpu_addresses.projection_camera_model_transform, false,
            Matrix.flatten_2D_to_1D(PCM.transposed()));
    }

    shared_glsl_code() {
        // ********* SHARED CODE, INCLUDED IN BOTH SHADERS *********
        return `
        precision mediump float;
        varying vec4 point_position;
        varying vec4 center;
        `;
    }

    vertex_glsl_code() {
        // ********* VERTEX SHADER *********
        // TODO:  Complete the main function of the vertex shader (Extra Credit Part II).
        return this.shared_glsl_code() + `
        attribute vec3 position;
        uniform mat4 model_transform;
        uniform mat4 projection_camera_model_transform;
        
        void main(){
          
        }`;
    }

    fragment_glsl_code() {
        // ********* FRAGMENT SHADER *********
        // TODO:  Complete the main function of the fragment shader (Extra Credit Part II).
        return this.shared_glsl_code() + `
        void main(){
          
        }`;
    }
}


// TODO: Leftover Collison code

// Draw walls
//         var i;
//         var j;
//         var map_maze = new Array(this.map_size);
//         for (i = 0; i < this.map_size; i++){
//             map_maze[i] = new Array(this.map_size);
//             for (j = 0; j < this.map_size; j++) {
//                 if (i == 0 || i == (this.map_size - 1) || j == 0 || j == (this.map_size - 1)) {
//                     map_maze[i][j] = "X";
//                 }
//                 else {
//                     map_maze[i][j] = "O";
//                 }
//                 // TODO: define other parts of the mazer here
//             }

//         }
        
//         // TODO: store cube coords
//         var wall_dict = [];
       
//         for (i = 0; i < this.map_size; i++){
//             for (j = 0; j < this.map_size; j++) {
//                 //if ( (i % 2 == 1) || (j % 2 == 1 ) ) { continue; }
//                 let model_transform_wall = Mat4.identity();
//                 model_transform_wall = model_transform_wall.times(Mat4.scale(0.5, 0.5, 0.5));

//                 let offset = this.map_size / 2;

//                 model_transform_wall = model_transform_wall.times(Mat4.translation(offset - i, -1, offset - j));

//                 if (map_maze[i][j] == "X") {
//                     // save box coords
//                     let box_center_3 = model_transform_wall.times( vec4( 0,0,0,1 ) ).to3();
//                     let box_center = (box_center_3[0], box_center_3[1]);
//                     console.log(box_center_3)
//                     let box_LL = (box_center_3[0] - 0.5, box_center_3[1] - 0.5);
//                     let box_LR = (box_center_3[0] - 0.5, box_center_3[1] + 0.5);
//                     let box_UL = (box_center_3[0] + 0.5, box_center_3[1] - 0.5);
//                     let box_UR = (box_center_3[0] + 0.5, box_center_3[1] + 0.5);

//                     wall_dict.push({center : box_center, LL : box_LL, LR : box_LR, UL : box_UL, UR : box_UR});


//                     var wh;  
//                     for ( wh = 0; wh < this.wall_height; wh++) {

//                         this.shapes.box.draw(context, program_state, model_transform_wall, this.materials.test);
//                         model_transform_wall = model_transform_wall.times(Mat4.translation(0, 1, 0));
//                         this.shapes.box.draw(context, program_state, model_transform_wall, this.materials.test);
//                     }                    
//                 }
//             }
//         }

//         //console.log(wall_dict);
        


//         // TODO: Collision
//         let cirlce_collider_center = vec4(program_state.camera_transform[0][3], this.height, program_state.camera_transform[2][3], 1);
//         let cirlce_collider_rad = 1;