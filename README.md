# tiny-graphics.js
By: Keiran Glynn (105364966)
	Eugene Choi (905368197)


1. Introduciton and Implementation:
  Our project is an endless world simulation, in which the player can walk 
through a procedurally generated world. While deciding what to create for this
assingment, our team agreed that the previous projects in this class had all
been small scenes that had limited scale. Since it runs in the browser, 
Tiny Graphics seems to be a big limiter. 

  Some of Our Unique Features:
  	1. Infinite Plane: The infinite plane in our case was our ground. We created a ground that expands forever by generating chunks
  					   in front of the player as they move in any x,z, or xz direction. This gives the player no limitations, allowing them to explore different biomes that are randomly selected based on a division of the player's current coordinate and the size a single ground chunk which then has it's returned value floored to signify
  					   which biome to generate.

  	2. Spawning of Trees: Each biome is mapped to have one of three types of trees to generate. The trees are determined based on 						division (similar to determining the biome). The determined trees are found from a dictionary that we 						  initialized with the tree and leaf offsets and the scales for the leaves. After being chosen, the 						  trees are spawed into the biome taking on the colors that are provided in the unique biome dictionary. 

  	3. Blender Objects: For the all the trees and leaves, we utilized the free open-source 3D computer graphics software toolset 					 called Blender to hand create them. Using the blender objects gave much more shape and uniqueness to our
  						trees and leaves as opposed to generating them using the provided primitive shapes.
 
Movement:
	w -> forward
	s -> backward
	a -> left
	d -> right
	, -> rotate left
	. -> rotate right

(Removed degrees of freedom to the y plane which would have required us to do collision detection. We have all degrees of
freedom in movement on the xz plane.)


2. Advanced Features:
The advanced features that we chose was bump mapping.


3. Referneces:
a. https://apoorvaj.io/exploring-bump-mapping-with-webgl/
b. http://shadedrelief.com/3D_Terrain_Maps/3dterrainmapsbum.html
c. https://conceptartempire.com/blender-modeling-tutorials/
d. https://eng.libretexts.org/Bookshelves/Computer_Science/
    Book%3A_Introduction_to_Computer_Graphics_(Eck)/
    07%3A_3D_Graphics_with_WebGL/7.03%3A_Textures
e. https://apoorvaj.io/exploring-bump-mapping-with-webgl/