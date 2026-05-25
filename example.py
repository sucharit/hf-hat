# This is an example file for the hf-hat program.

# Copyright (C) 2015 Sucharit Sarkar.
# Contact: sucharit@math.princeton.edu

# Copying and distribution of this file, with or without modification,
# are permitted in any medium without royalty provided the copyright
# notice and this notice are preserved.

##############

from hf_hat import HeegaardDiagram, branched_double

# Declare a Heegaard diagram as HeegaardDiagram(boundary_intersections, num_pointed_regions)
# num_pointed_regions is the number of pointed regions in the Heegaard diagram
# (through which domains won't pass).
# Then number the regions (both pointed and unpointed) starting at 0, so that
# all the pointed regions are numbered last. Also number the intersections
# between alpha and beta circles starting at 0 in some arbitrary order.
# boundary_intersections is a list whose i-th element is a list of intersection
# points that lie on the boundary of the i-th region (in the same order as per
# the boundary orientation of that region), so that the first two points are on
# some alpha circle.
# (Currently, only works if none of the alpha or beta circles have exactly two
# intersection points and if there are no unpointed periodic domains, as in the
# case of rational homology spheres with same number of alpha curves as the
# Heegaard surface genus.)

print("Present restrictions:\n"
      " 1. None of the alpha or beta circles may have exactly two "
      "intersection points.\n"
      " 2. There are no unpointed periodic domains (example: rational homology "
      "spheres with same number of alpha curves as the Heegaard surface genus).")

S234 = HeegaardDiagram([
        [0,1,10,9],
        [9,8,6,7,4,0],
        [5,6,8,7,0,4],
        [4,3,10,5],
        [1,0,7,6],
        [2,1,6,5],
        [3,4,7,8],
        [2,3,8,9],
        [1,2,9,10,3,2,5,10]
        ], 1)  # Heegaard diagram for Sigma(2,3,4).

# The following is a list of variables in the HeegaardDiagram class.
##############
print(S234.boundary_intersections)  # list: i-th element = boundary intersections of region i
print(S234.regions)                 # the regions
print(S234.regions_un)              # the unpointed regions
print(S234.intersections)           # the intersection points between alpha and beta circles
print(S234.is_nice)                 # True or False
print(S234.euler_measures_2)        # list: i-th element = 2 * Euler measure of region i
print(S234.euler_measures_2_un)     # numpy array: unpointed-region version
print(S234.boundary_mat)            # boundary_mat[i][j]: coeff of d(d(region i)|_alpha) at point j
print(S234.boundary_mat_un)         # numpy matrix: unpointed-region version
print(S234.point_measures_4)        # point_measures_4[i][j] = 4 * point measure of region i at j
print(S234.point_measures_4_un)     # numpy matrix: unpointed-region version
print(S234.alphas)                  # the alpha circles
print(S234.betas)                   # the beta circles
print(S234.intersections_on_alphas) # list: i-th element = ordered intersections on alpha_i
print(S234.intersections_on_betas)  # ditto for beta
print(S234.intersection_incidence)  # intersection_incidence[i] = [alpha_idx, beta_idx, sign]
print(S234.intersection_matrix)     # numpy matrix: (a,b) entry = intersection number alpha_a . beta_b
# region_graph_alpha, region_graph_beta: _SimpleMultigraph objects
print(S234.generators)              # the CF-hat generators (by index)
print(S234.generator_reps)          # generator_reps[i] = tuple of intersections (one per alpha)

# The following variables are not initialized until generate_domains() is called.
##############
# S234.SpinC_structures     -- the SpinC structures
# S234.SpinC                -- SpinC[i] = SpinC structure of i-th generator
# S234.genSpinC             -- genSpinC[i] = list of generators in i-th SpinC structure
# S234.domains_stored       -- domains_stored[i][j] = (maslov_index, domain_vector) or None
# S234.abs_gr               -- absolute gradings
# S234.gr_min, S234.gr_max
# S234.grading_spread       -- grading_spread[gr] = list of generators with abs_gr=gr

# S234.chain_complex is not initialized until generate_complexes() is called.
##############

# We run the following functions with i=10, j=0, gr=0.
[i, j, gr] = [10, 0, 0]
##############
S234.generate_domains()      # initializes SpinC info, domains_stored, abs_gr, etc.
S234.can_contribute(i, j)    # True if domain i→j has Maslov index 1 and is positive
S234.does_contribute(i, j)   # True if we know d[i→j] = 1; False = no knowledge
S234.domain_type(i, j)       # (Euler char, num boundary components) if can_contribute
S234.set_abs_gr(i, gr)       # set abs_gr of i-th generator to gr

try:
    S234.generate_complexes()   # initializes chain_complex
    print(S234.compute_homology())  # HF-hat: list of {grading: rank} dicts per SpinC
except Exception as e:
    print("generate_complexes / compute_homology raised:", e)

##############
print(repr(S234))                 # prints details about the Heegaard diagram
S234.print_differentials()        # prints differential structure
S234.pretty_print()               # displays chain complex graphically (interactive)


# Sometimes there is a Z/2-action on the Heegaard diagram.
# Declare as HeegaardDiagram(boundary_intersections, num_pointed_regions, image_of_intersections)
# image_of_intersections[i] = Z/2-action image of the i-th intersection point.
S237 = HeegaardDiagram([
        [3,4,7,8,10,11],
        [8,5,13,14,9,10],
        [5,6,16,13],
        [14,15,3,2,12,9],
        [1,2,1,0,5,8],
        [0,4,9,12,6,5],
        [12,11,15,14,7,6],
        [11,10,16,15],
        [14,13,0,1,8,7],
        [2,1,2,3,11,12],
        [15,16,6,7,4,0,13,16,10,9,4,3]
        ], 1,
        [3,2,1,0,4,11,10,9,12,7,6,5,8,15,14,13,16])  # Sigma(2,3,7) with Z/2-action.

# Extra variables available when a Z/2-action is given.
##############
print(S237.there_is_action)           # True
print(S237.image_of_intersections)    # list: image of each intersection under the action
# After generate_domains():
# S237.image_of_generators          -- image of each generator
# S237.image_of_SpinC_structures    -- image of each SpinC structure
# After generate_complexes():
# S237.mapping_cone_complex         -- mapping cone of (Id + action) per SpinC structure

try:
    print(S237.compute_mc_homology())  # homology of mapping cone of (Id + action)
except Exception as e:
    print("compute_mc_homology raised:", e)

# The program can also compute double branched covers along knots.
# (Requires the knot to be Z/2-nullhomologous.)
# branched_double(H, num_pointed_regions)
#   H must have exactly two pointed regions.
#   num_pointed_regions=1: one basepoint on preimage of last region.
#   num_pointed_regions=2: basepoints on preimages of both pointed regions.
trefoil = HeegaardDiagram([[0,1,2,0,2,1,0,2],[1,0],[1,2]], 2)
L31 = branched_double(trefoil, 1)


##############
# pretty_print displays the CF-hat chain complex graphically.
# Red arrows: known coefficient 1. Blue: unknown. Green: zig-zag from cancellations.
# Parameters:
#   output_file   -- path to save (pdf/png/…); None = display interactively
#   interchanges  -- list of (g, h) generator pairs to swap horizontally
#   cancellations -- list of (a, b) pairs with known d[a→b]=1 to cancel
#   relevant_SpinC -- list of SpinC structure indices to show; 'all' = show all

S237.pretty_print(output_file="S237_0.pdf")

# You may make the diagram prettier via interchanges.
S237.pretty_print(output_file="S237_1.pdf",
                  interchanges=[(12,9),(13,1),(16,5),(14,0),(17,4),(15,6),
                                (6,18),(6,0),(15,3),(15,14)])

# You may also simplify the complex by cancellations.
S237.pretty_print(output_file="S237_2.pdf",
                  interchanges=[(12,9),(13,1),(16,5),(14,0),(17,4),(15,6),
                                (6,18),(6,0),(15,3),(15,14)],
                  cancellations=[(6,4),(18,23),(0,11),(15,17),(3,21),
                                 (14,10),(5,1),(16,13)])

# When a Z/2-action is given and relevant_SpinC is fixed under the action,
# pretty_print tries to draw equivariantly.  Tweak and cancel equivariantly
# in that case if you want the Z/2 symmetry to remain.
