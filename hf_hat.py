## Copyright (C) 2015 Sucharit Sarkar.
## Contact: sucharit@math.princeton.edu
## Pure Python port (C) 2026.

## This file is part of hf-hat.

## hf-hat is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.

## hf-hat is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
## General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with hf-hat; see COPYING.  If not, see
## <http://www.gnu.org/licenses/>.

import itertools
from fractions import Fraction

import numpy as np
import matplotlib.pyplot as plt

from gf2 import GF2Matrix


# ---------------------------------------------------------------------------
# Helpers replacing SageMath builtins
# ---------------------------------------------------------------------------

def _flatten(lst):
    """Flatten one level of nesting."""
    return [x for sublist in lst for x in sublist]


def _solve_left_ZZ(A, b):
    """Solve x @ A = b over Z (exact, using Fraction arithmetic).

    A : numpy int array of shape (nrows, ncols)
    b : numpy int array of length ncols

    Returns x as numpy int array of length nrows.
    Raises ValueError if no rational solution exists.
    Raises TypeError  if the unique rational solution is not integral.
    """
    nrows, ncols = A.shape
    # Equivalent to solving A^T @ x = b  (ncols equations, nrows unknowns).
    # Build augmented matrix [A^T | b] over Q.
    aug = [
        [Fraction(int(A[i, j])) for i in range(nrows)] + [Fraction(int(b[j]))]
        for j in range(ncols)
    ]

    pivot_row = 0
    pivot_cols = []
    for col in range(nrows):
        found = -1
        for row in range(pivot_row, ncols):
            if aug[row][col] != 0:
                found = row
                break
        if found == -1:
            continue
        aug[pivot_row], aug[found] = aug[found], aug[pivot_row]
        piv = aug[pivot_row][col]
        aug[pivot_row] = [v / piv for v in aug[pivot_row]]
        for row in range(ncols):
            if row != pivot_row and aug[row][col] != 0:
                f = aug[row][col]
                aug[row] = [aug[row][k] - f * aug[pivot_row][k]
                            for k in range(nrows + 1)]
        pivot_cols.append(col)
        pivot_row += 1

    # Consistency check
    for row in range(pivot_row, ncols):
        if aug[row][nrows] != 0:
            raise ValueError("No solution over Q")

    # Extract solution (free variables = 0)
    x = [Fraction(0)] * nrows
    for i, col in enumerate(pivot_cols):
        x[col] = aug[i][nrows]

    # Integrality check
    x_int = []
    for v in x:
        if v.denominator != 1:
            raise TypeError("Solution is not integral")
        x_int.append(int(v))

    return np.array(x_int, dtype=int)


# ---------------------------------------------------------------------------
# Minimal multigraph (replaces SageMath Graph)
# ---------------------------------------------------------------------------

class _SimpleMultigraph:
    """Undirected multigraph with labeled edges; supports BFS shortest path."""

    def __init__(self, n):
        self._edges = []          # list of (u, v, label)
        self._adj = [[] for _ in range(n)]

    def add_edge(self, edge_spec):
        """edge_spec is [u, v, label]."""
        u, v, label = edge_spec[0], edge_spec[1], edge_spec[2]
        self._edges.append((u, v, label))
        self._adj[u].append(v)
        if u != v:
            self._adj[v].append(u)

    def edges(self, labels=True):
        return list(self._edges)

    def shortest_path(self, u, v):
        """BFS shortest path from u to v; returns list of vertex indices."""
        if u == v:
            return [u]
        prev = {u: None}
        queue = [u]
        while queue:
            curr = queue.pop(0)
            for nxt in self._adj[curr]:
                if nxt not in prev:
                    prev[nxt] = curr
                    if nxt == v:
                        path = []
                        node = v
                        while node is not None:
                            path.append(node)
                            node = prev[node]
                        return list(reversed(path))
                    queue.append(nxt)
        return []


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class HeegaardDiagram():

    def __init__(self, boundary_intersections, num_pointed_regions,
                 image_of_intersections=None):
        """
        Number all the regions (complement of alpha,beta), starting at
        0, so that all the unpointed regions come first. Number all
        the intersection points between alpha,beta, again starting at
        0 (the numbering is arbitrary).

        The boundary_intersections is a list whose i-th element is a list of
        intersection points that appear on the boundary of the i-th
        region, according to the boundary orientation, so that the
        part of the boundary joining the first two points lie on an
        alpha circle. (We are implicitly assuming that all regions are
        planar with one boundary (which can intersect itself).)

        The num_pointed_regions is the number of pointed regions
        (always the last few).

        Sometimes, there is a Z/2-action. The program can work with a
        single such Z/2-action. If there is a Z/2-action, input
        image_of_intersections as list whose i-th element is the
        Z/2-action image of the i-th intersection point. (The default
        is None.)
        """

        # Basic initialization: convert each region boundary to its
        # lexicographically smallest cyclic rotation.
        self.boundary_intersections = []
        for intersections_list in boundary_intersections:
            smallest_indices = [2*ind_i for ind_i, i
                                in enumerate(intersections_list[0::2])
                                if i == min(intersections_list[0::2])]
            next_intersections = [intersections_list[ind_i+1]
                                  for ind_i in smallest_indices]
            smallest_index = next(
                ind_i for ind_i in smallest_indices
                if intersections_list[ind_i+1] == min(next_intersections))
            self.boundary_intersections.append(
                intersections_list[smallest_index:]
                + intersections_list[:smallest_index])

        if image_of_intersections is not None:
            self.there_is_action = True
            self.image_of_intersections = image_of_intersections
        else:
            self.there_is_action = False

        self.regions = list(range(len(self.boundary_intersections)))
        self.regions_un = list(range(
            len(self.boundary_intersections) - num_pointed_regions))
        self.intersections = list(range(
            1 + max(_flatten(self.boundary_intersections))))

        # Euler measures of the regions, times 2.
        self.euler_measures_2 = [
            2 - len(self.boundary_intersections[R]) // 2
            for R in self.regions]
        self.euler_measures_2_un = np.array(
            self.euler_measures_2[:len(self.regions_un)], dtype=int)
        self.is_nice = bool(min(self.euler_measures_2_un) > -1)

        # boundary_mat[i][j]: coefficient of d(d(region_i)|_alpha) at point j.
        self.boundary_mat = [
            [-self.boundary_intersections[R][0::2].count(p)
             + self.boundary_intersections[R][1::2].count(p)
             for p in self.intersections]
            for R in self.regions]
        self.boundary_mat_un = np.array(
            self.boundary_mat[:len(self.regions_un)], dtype=int)

        # point_measures_4[i][j]: 4 * (point measure of region i at point j).
        self.point_measures_4 = [
            [self.boundary_intersections[R].count(p)
             for p in self.intersections]
            for R in self.regions]
        self.point_measures_4_un = np.array(
            self.point_measures_4[:len(self.regions_un)], dtype=int)

        # Recover alpha circles by following even-indexed -> odd-indexed arcs.
        self.intersections_on_alphas = []
        while len(_flatten(self.intersections_on_alphas)) < len(self.intersections):
            start_p = next(p for p in self.intersections
                           if p not in _flatten(self.intersections_on_alphas))
            new_circle = []
            curr_p = start_p
            while curr_p != start_p or new_circle == []:
                new_circle.append(curr_p)
                found_next_point = False
                regions_with_curr_point = [
                    (R, ind_p)
                    for R in self.regions
                    for ind_p in range(len(self.boundary_intersections[R]))
                    if curr_p == self.boundary_intersections[R][ind_p]]
                for (R, ind_p) in regions_with_curr_point:
                    if not found_next_point:
                        if (ind_p % 2 == 0
                                and self.boundary_intersections[R][ind_p+1]
                                not in new_circle):
                            found_next_point = True
                            curr_p = self.boundary_intersections[R][ind_p+1]
                        elif (ind_p % 2 == 1
                              and self.boundary_intersections[R][ind_p-1]
                              not in new_circle):
                            found_next_point = True
                            curr_p = self.boundary_intersections[R][ind_p-1]
                if not found_next_point:
                    curr_p = start_p
            self.intersections_on_alphas.append(new_circle)

        # Recover beta circles by following odd-indexed -> even-indexed arcs.
        self.intersections_on_betas = []
        while len(_flatten(self.intersections_on_betas)) < len(self.intersections):
            start_p = next(p for p in self.intersections
                           if p not in _flatten(self.intersections_on_betas))
            new_circle = []
            curr_p = start_p
            while curr_p != start_p or new_circle == []:
                new_circle.append(curr_p)
                found_next_point = False
                regions_with_curr_point = [
                    (R, ind_p)
                    for R in self.regions
                    for ind_p in range(len(self.boundary_intersections[R]))
                    if curr_p == self.boundary_intersections[R][ind_p]]
                for (R, ind_p) in regions_with_curr_point:
                    if not found_next_point:
                        if (ind_p % 2 == 0
                                and self.boundary_intersections[R][ind_p-1]
                                not in new_circle):
                            found_next_point = True
                            curr_p = self.boundary_intersections[R][ind_p-1]
                        elif (ind_p % 2 == 1
                              and self.boundary_intersections[R][
                                  ind_p+1-len(self.boundary_intersections[R])]
                              not in new_circle):
                            found_next_point = True
                            curr_p = self.boundary_intersections[R][
                                ind_p+1-len(self.boundary_intersections[R])]
                if not found_next_point:
                    curr_p = start_p
            self.intersections_on_betas.append(new_circle)

        self.alphas = list(range(len(self.intersections_on_alphas)))
        self.betas  = list(range(len(self.intersections_on_betas)))

        if sorted(_flatten(self.intersections_on_alphas)) != self.intersections:
            raise Exception("Alpha circles don't contain all the intersections")
        if sorted(_flatten(self.intersections_on_betas)) != self.intersections:
            raise Exception("Beta circles don't contain all the intersections")

        # intersection_incidence[p] = [alpha_index, beta_index]
        # (and later appended: intersection_number)
        self.intersection_incidence = [
            [next(a for a in self.alphas
                  if p in self.intersections_on_alphas[a]),
             next(b for b in self.betas
                  if p in self.intersections_on_betas[b])]
            for p in self.intersections]

        for p in self.intersections:
            try:
                [a, b] = self.intersection_incidence[p]  # ValueError if already length 3
                if (len(self.intersections_on_alphas[a]) <= 2
                        or len(self.intersections_on_betas[b]) <= 2):
                    raise Exception("Failed to orient alpha or beta circle")
                else:
                    a_pind = self.intersections_on_alphas[a].index(p)
                    b_pind = self.intersections_on_betas[b].index(p)
                    (R, ind_p) = next(
                        (R, 2*k)
                        for R in self.regions
                        for k in range(len(self.boundary_intersections[R]) // 2)
                        if self.boundary_intersections[R][2*k] == p)
                    prev_b = self.boundary_intersections[R][ind_p-1]
                    next_a = self.boundary_intersections[R][ind_p+1]
                    intersection_number = -1  # assuming a: p→next_a, b: prev_b→p

                    if self.intersections_on_alphas[a][a_pind-1] == next_a:
                        intersection_number = -intersection_number
                    if self.intersections_on_betas[b][b_pind-1] != prev_b:
                        intersection_number = -intersection_number

                    self.intersection_incidence[p].append(intersection_number)
            except ValueError:
                pass  # intersection number already determined

        self.intersection_matrix = np.zeros(
            (len(self.alphas), len(self.betas)), dtype=int)
        for a in self.alphas:
            for b in self.betas:
                self.intersection_matrix[a, b] = sum(
                    n for [i, j, n] in self.intersection_incidence
                    if a == i and b == j)

        # region_graph_alpha: vertices = regions, edges = alpha arcs.
        self.region_graph_alpha = _SimpleMultigraph(len(self.regions))
        for a in self.alphas:
            for ind_p, p in enumerate(self.intersections_on_alphas[a]):
                q = self.intersections_on_alphas[a][
                    ind_p+1-len(self.intersections_on_alphas[a])]
                regions_with_pq_1 = [
                    R for R in self.regions
                    for ind_foo, foo in enumerate(
                        self.boundary_intersections[R][0::2])
                    if [foo, self.boundary_intersections[R][2*ind_foo+1]]
                    == [p, q]]
                regions_with_pq_2 = [
                    R for R in self.regions
                    for ind_foo, foo in enumerate(
                        self.boundary_intersections[R][0::2])
                    if [foo, self.boundary_intersections[R][2*ind_foo+1]]
                    == [q, p]]
                if len(regions_with_pq_1) != 1 or len(regions_with_pq_2) != 1:
                    raise Exception(
                        "Each alpha arc has two adjacent regions on two sides.")
                self.region_graph_alpha.add_edge(
                    regions_with_pq_1 + regions_with_pq_2 + [(a, p)])

        # region_graph_beta: vertices = regions, edges = beta arcs.
        self.region_graph_beta = _SimpleMultigraph(len(self.regions))
        for b in self.betas:
            for ind_p, p in enumerate(self.intersections_on_betas[b]):
                q = self.intersections_on_betas[b][
                    ind_p+1-len(self.intersections_on_betas[b])]
                regions_with_pq_1 = [
                    R for R in self.regions
                    for ind_foo, foo in enumerate(
                        self.boundary_intersections[R][0::2])
                    if [foo, self.boundary_intersections[R][2*ind_foo-1]]
                    == [p, q]]
                regions_with_pq_2 = [
                    R for R in self.regions
                    for ind_foo, foo in enumerate(
                        self.boundary_intersections[R][0::2])
                    if [foo, self.boundary_intersections[R][2*ind_foo-1]]
                    == [q, p]]
                if len(regions_with_pq_1) != 1 or len(regions_with_pq_2) != 1:
                    raise Exception(
                        "Each beta arc has two adjacent regions on two sides.")
                self.region_graph_beta.add_edge(
                    regions_with_pq_1 + regions_with_pq_2 + [(b, p)])

        # Error checks.
        if np.any(
                np.ones(len(self.regions), dtype=int)
                @ np.array(self.point_measures_4, dtype=int)
                != np.full(len(self.intersections), 4, dtype=int)):
            raise Exception("Every intersection should have four corners.")
        if len(self.alphas) != len(self.betas):
            raise Exception("Require same number of alpha and beta circles.")

        if self.there_is_action:
            for p in self.intersections:
                if self.image_of_intersections[
                        self.image_of_intersections[p]] != p:
                    raise Exception("Not an involution.")
            for R in self.regions:
                image_of_R = [self.image_of_intersections[p]
                               for p in self.boundary_intersections[R]]
                si = [2*ind_i for ind_i, i
                      in enumerate(image_of_R[0::2])
                      if i == min(image_of_R[0::2])]
                ni = [image_of_R[k+1] for k in si]
                sk = next(k for k in si if image_of_R[k+1] == min(ni))
                image_of_R = image_of_R[sk:] + image_of_R[:sk]
                if image_of_R not in self.boundary_intersections:
                    raise Exception("The Z/2-action is not a valid action.")

        # Generators: tuples (p_0, p_1, ...) where p_i is on alpha_i.
        self.generator_reps = []
        for permu in itertools.permutations(self.alphas):
            possibilities = [
                [p for p in self.intersections_on_alphas[a]
                 if self.intersection_incidence[p][1] == permu[a]]
                for a in self.alphas]
            self.generator_reps += list(itertools.product(*possibilities))
        self.generators = list(range(len(self.generator_reps)))

        self.generators_dict = {
            tuple(self.generator_reps[g]): g for g in self.generators}

        if self.there_is_action:
            self.image_of_generators = []
            for g in self.generators:
                image = [self.image_of_intersections[self.generator_reps[g][a]]
                         for a in self.alphas]
                image = tuple(
                    next(p for p in image
                         if self.intersection_incidence[p][0] == a)
                    for a in self.alphas)
                self.image_of_generators.append(self.generators_dict[image])

        # Variables initialized lazily by generate_domains() / generate_complexes().
        self.SpinC = "Not yet initialized"
        self.SpinC_structures = "Not yet initialized"
        self.genSpinC = "Not yet initialized"
        if self.there_is_action:
            self.image_of_SpinC_structures = "Not yet initialized"
        self.domains_stored = "Not yet initialized"
        self.abs_gr = "Not yet initialized"
        self.gr_min = "Not yet initialized"
        self.gr_max = "Not yet initialized"
        self.grading_spread = "Not yet initialized"
        self.fully_initialized = False

        self.chain_complex = "Not yet initialized"
        if self.there_is_action:
            self.mapping_cone_complex = "Not yet initialized"
        self.chain_complexes_initialized = False


    def __repr__(self):
        return (
            "Heegaard diagram with " + repr(len(self.alphas))
            + " alpha and beta circles intersecting each other in "
            + repr(len(self.intersections)) + " intersection points. "
            + "The intersection points on the alpha circles are "
            + repr(self.intersections_on_alphas)
            + " and the intersection points on the beta circles are "
            + repr(self.intersections_on_betas)
            + ". There are " + repr(len(self.regions_un))
            + " unpointed regions, and the intersection points appearing "
            + "on their boundaries are "
            + repr(self.boundary_intersections[:len(self.regions_un)]) + ".")


    def find_domain(self, initial, final):
        """Find a domain D from initial to final; return (Maslov_index(D), D)."""
        target_vect = np.zeros(len(self.intersections), dtype=int)
        target_vect_abs = np.zeros(len(self.intersections), dtype=int)
        for p in self.intersections:
            target_vect[p] = (self.generator_reps[final].count(p)
                              - self.generator_reps[initial].count(p))
            target_vect_abs[p] = (self.generator_reps[final].count(p)
                                  + self.generator_reps[initial].count(p))

        answer = _solve_left_ZZ(self.boundary_mat_un, target_vect)
        maslov = int(
            2 * int(answer @ self.euler_measures_2_un)
            + int(answer @ self.point_measures_4_un @ target_vect_abs))
        if maslov % 4 != 0:
            raise Exception('Maslov index not an integer!')
        maslov = maslov // 4

        return (maslov, answer)


    def generate_domains(self):
        """Fill domains_stored and all SpinC / grading information."""
        if self.fully_initialized:
            return True

        self.domains_stored = []
        self.SpinC = [0] + (len(self.generators)-1) * [None]
        self.abs_gr = len(self.generators) * [None]
        self.grading_spread = dict()

        for initial in self.generators:
            self.domains_stored.append([])
            for final in self.generators:
                S = self.SpinC[final]
                if final < initial:
                    if self.domains_stored[final][initial] is not None:
                        (m, D) = self.domains_stored[final][initial]
                        self.domains_stored[initial].append((-m, -D))
                    else:
                        self.domains_stored[initial].append(None)
                elif final == initial:
                    self.domains_stored[initial].append(
                        (0, np.zeros(len(self.regions_un), dtype=int)))
                else:
                    if S is None:
                        try:
                            (m, D) = self.find_domain(initial, final)
                            self.domains_stored[initial].append((m, D))
                            self.SpinC[final] = self.SpinC[initial]
                        except (ValueError, TypeError):
                            if final == initial + 1:
                                self.SpinC[final] = max(
                                    s for s in self.SpinC if s is not None) + 1
                            self.domains_stored[initial].append(None)
                    elif S == self.SpinC[initial]:
                        first = next(g for g in self.generators
                                     if self.SpinC[g] == S)
                        (m1, D1) = self.domains_stored[first][initial]
                        (m2, D2) = self.domains_stored[first][final]
                        self.domains_stored[initial].append((m2-m1, D2-D1))
                    else:
                        self.domains_stored[initial].append(None)

        self.SpinC_structures = list(range(max(self.SpinC) + 1))
        self.genSpinC = [
            [g for g in self.generators if self.SpinC[g] == S]
            for S in self.SpinC_structures]
        if self.there_is_action:
            self.image_of_SpinC_structures = [
                self.SpinC[self.image_of_generators[self.genSpinC[S][0]]]
                for S in self.SpinC_structures]

        for S in self.SpinC_structures:
            base_gen = self.genSpinC[S][0]
            self.abs_gr[base_gen] = 0
            if self.there_is_action:
                if self.image_of_SpinC_structures[S] < S:
                    self.abs_gr[base_gen] = self.abs_gr[
                        self.image_of_generators[base_gen]]
                elif self.image_of_SpinC_structures[S] == S:
                    if self.domains_stored[
                            self.image_of_generators[base_gen]][base_gen][0] != 0:
                        raise Exception(
                            "The action does not respect gradings.")
            for g in self.genSpinC[S]:
                self.abs_gr[g] = (self.abs_gr[base_gen]
                                  + self.domains_stored[g][base_gen][0])
        self.gr_min = min(self.abs_gr)
        self.gr_max = max(self.abs_gr)
        for gr in set(self.abs_gr):
            self.grading_spread[gr] = [g for g in self.generators
                                        if self.abs_gr[g] == gr]

        self.fully_initialized = True


    def set_abs_gr(self, base_gen, gr):
        """Set abs_gr of generator base_gen to value gr."""
        if not self.fully_initialized:
            self.generate_domains()

        change = gr - self.abs_gr[base_gen]
        S = self.SpinC[base_gen]
        if self.there_is_action:
            T = self.image_of_SpinC_structures[S]
        if self.there_is_action and T != S:
            generators_to_change = self.genSpinC[S] + self.genSpinC[T]
        else:
            generators_to_change = self.genSpinC[S]
        for g in generators_to_change:
            self.abs_gr[g] += change

        self.gr_min = min(self.abs_gr)
        self.gr_max = max(self.abs_gr)
        self.grading_spread = dict()
        for gr in set(self.abs_gr):
            self.grading_spread[gr] = [g for g in self.generators
                                        if self.abs_gr[g] == gr]

        if self.chain_complexes_initialized:
            self.chain_complex[S] = {
                gr2+change: self.chain_complex[S][gr2]
                for gr2 in self.chain_complex[S]}
            if self.there_is_action:
                if T != S:
                    self.chain_complex[T] = {
                        gr2+change: self.chain_complex[T][gr2]
                        for gr2 in self.chain_complex[T]}
                else:
                    self.mapping_cone_complex[S] = {
                        gr2+change: self.mapping_cone_complex[S][gr2]
                        for gr2 in self.mapping_cone_complex[S]}


    def can_contribute(self, i, j):
        """Return True if domain from generator i to j has Maslov index 1 and is positive."""
        if not self.fully_initialized:
            self.generate_domains()
        try:
            (m, D) = self.domains_stored[i][j]
            return m == 1 and not np.any(D < 0)
        except TypeError:
            return False


    def domain_type(self, i, j):
        """Analyse domain from i to j (only if can_contribute). Return (euler_char, num_bdy_comps)."""
        if not self.can_contribute(i, j):
            raise Exception("No positive Maslov index 1 domain")
        if self.is_nice:
            return (1, 1)

        (m, D) = self.domains_stored[i][j]
        e2 = int(D @ self.euler_measures_2_un)
        iota = m - e2

        Dp = D @ self.point_measures_4_un
        alpha_coefficients = [
            int(sum(Dp[p] for p in self.intersections_on_alphas[a]))
            for a in self.alphas]
        num_trivial = alpha_coefficients.count(0)
        euler = len(self.alphas) - num_trivial - iota

        bdy_comps = [None] * len(self.alphas)
        while None in bdy_comps:
            first_alpha = bdy_comps.index(None)
            assigned = [v for v in bdy_comps if v is not None]
            bdy_comps[first_alpha] = (max(assigned) + 1) if assigned else 0

            current_alpha = first_alpha
            reached_end_of_loop = False
            while not reached_end_of_loop:
                next_alpha = next(
                    a for a in self.alphas
                    if self.intersection_incidence[
                           self.generator_reps[j][current_alpha]][1]
                    == self.intersection_incidence[
                           self.generator_reps[i][a]][1])
                bdy_comps[next_alpha] = bdy_comps[current_alpha]
                current_alpha = next_alpha
                if current_alpha == first_alpha:
                    reached_end_of_loop = True

        return (euler, max(bdy_comps) + 1 - num_trivial)


    def does_contribute(self, i, j):
        """Return True if we know the domain from i to j contributes to the differential."""
        (e, b) = self.domain_type(i, j)
        if (e, b) == (1, 1):
            return True
        return False


    def generate_complexes(self):
        """Build chain_complex (and mapping_cone_complex if Z/2-action exists)."""
        if self.chain_complexes_initialized:
            return True

        if not self.fully_initialized:
            self.generate_domains()

        if not self.is_nice:
            for g in self.generators:
                for h in self.generators:
                    if self.can_contribute(g, h) and not self.does_contribute(g, h):
                        raise Exception("Not enough data to generate complexes")

        self.chain_complex = []
        if self.there_is_action:
            self.mapping_cone_complex = []

        for S in self.SpinC_structures:
            chain_complex = dict()
            gradings_in_S = sorted(set(
                self.abs_gr[g] for g in self.genSpinC[S]))
            for gr in gradings_in_S:
                if (gr-1) in gradings_in_S:
                    generators_higher = [g for g in self.genSpinC[S]
                                         if self.abs_gr[g] == gr]
                    generators_lower  = [g for g in self.genSpinC[S]
                                         if self.abs_gr[g] == gr-1]
                    boundary_matrix = GF2Matrix(
                        len(generators_higher), len(generators_lower))
                    for ind_g, g in enumerate(generators_higher):
                        for ind_h, h in enumerate(generators_lower):
                            if self.can_contribute(g, h):
                                boundary_matrix[ind_g, ind_h] = 1
                    chain_complex[gr] = boundary_matrix
            self.chain_complex.append(chain_complex)

        if self.there_is_action:
            for S in self.SpinC_structures:
                if self.image_of_SpinC_structures[S] != S:
                    self.mapping_cone_complex.append(None)
                else:
                    mapping_cone_complex = dict()
                    gradings_in_S = sorted(set(
                        [self.abs_gr[g] for g in self.genSpinC[S]]
                        + [self.abs_gr[g]+1 for g in self.genSpinC[S]]))
                    for gr in gradings_in_S:
                        if (gr-1) in gradings_in_S:
                            generators_higher = (
                                [g for g in self.genSpinC[S]
                                 if self.abs_gr[g] == gr]
                                + [g for g in self.genSpinC[S]
                                   if self.abs_gr[g] == gr-1])
                            generators_lower = (
                                [g for g in self.genSpinC[S]
                                 if self.abs_gr[g] == gr-1]
                                + [g for g in self.genSpinC[S]
                                   if self.abs_gr[g] == gr-2])
                            mc_boundary_matrix = GF2Matrix(
                                len(generators_higher), len(generators_lower))
                            for ind_g, g in enumerate(generators_higher):
                                for ind_h, h in enumerate(generators_lower):
                                    if self.can_contribute(g, h):
                                        mc_boundary_matrix[ind_g, ind_h] = 1
                                    if (self.image_of_generators[g] != g
                                            and (g == h
                                                 or self.image_of_generators[g]
                                                 == h)):
                                        mc_boundary_matrix[ind_g, ind_h] = 1
                            mapping_cone_complex[gr] = mc_boundary_matrix
                    self.mapping_cone_complex.append(mapping_cone_complex)

        self.chain_complexes_initialized = True


    def compute_homology(self):
        """Compute HF-hat homology. Return list of dicts {grading: rank} per SpinC."""
        if not self.chain_complexes_initialized:
            self.generate_complexes()

        homology = []
        for S in self.SpinC_structures:
            homology_in_S = dict()
            ranks_in_S = dict()
            gradings_in_S = sorted(set(
                self.abs_gr[g] for g in self.genSpinC[S]))
            for gr in self.chain_complex[S]:
                ranks_in_S[gr] = self.chain_complex[S][gr].rank()
            for gr in gradings_in_S:
                homology_in_S[gr] = len(
                    [g for g in self.genSpinC[S] if self.abs_gr[g] == gr])
                if (gr-1) in gradings_in_S:
                    homology_in_S[gr] -= ranks_in_S[gr]
                if (gr+1) in gradings_in_S:
                    homology_in_S[gr] -= ranks_in_S[gr+1]
            homology.append({gr: homology_in_S[gr]
                              for gr in homology_in_S
                              if homology_in_S[gr] != 0})
        return homology


    def compute_mc_homology(self):
        """Compute homology of the mapping cone of (Id + Z/2-action)."""
        if not self.there_is_action:
            raise Exception("No map to have a mapping cone complex")
        if not self.chain_complexes_initialized:
            self.generate_complexes()

        mc_homology = []
        for S in self.SpinC_structures:
            if self.mapping_cone_complex[S] is None:
                mc_homology.append(None)
            else:
                mc_homology_in_S = dict()
                ranks_in_S = dict()
                gradings_in_S = sorted(set(
                    [self.abs_gr[g] for g in self.genSpinC[S]]
                    + [self.abs_gr[g]+1 for g in self.genSpinC[S]]))
                for gr in self.mapping_cone_complex[S]:
                    ranks_in_S[gr] = self.mapping_cone_complex[S][gr].rank()
                for gr in gradings_in_S:
                    mc_homology_in_S[gr] = len(
                        [g for g in self.genSpinC[S] if self.abs_gr[g] == gr]
                        + [g for g in self.genSpinC[S]
                           if self.abs_gr[g] == gr-1])
                    if (gr-1) in gradings_in_S:
                        mc_homology_in_S[gr] -= ranks_in_S[gr]
                    if (gr+1) in gradings_in_S:
                        mc_homology_in_S[gr] -= ranks_in_S[gr+1]
                mc_homology.append({gr: mc_homology_in_S[gr]
                                    for gr in mc_homology_in_S
                                    if mc_homology_in_S[gr] != 0})
        return mc_homology


    def print_action(self):
        if self.there_is_action:
            for g in self.generators:
                print(repr(g) + " under the action maps to "
                      + repr(self.image_of_generators[g]))
        else:
            print("No action specified.")
        return None


    def print_differentials(self):
        if not self.fully_initialized:
            self.generate_domains()
        for S in self.SpinC_structures:
            print("In SpinC structure " + repr(S))
            for i in self.genSpinC[S]:
                targets = [j for j in self.genSpinC[S]
                           if self.can_contribute(i, j)]
                print(repr(i) + " in grading " + repr(self.abs_gr[i])
                      + " under the differential (possibly) maps to "
                      + repr(targets)
                      + " with domain types (Euler char, num of boundaries): "
                      + repr([self.domain_type(i, j) for j in targets]))
        return None


    def pretty_print(self, output_file=None, interchanges=None,
                     cancellations=None, relevant_SpinC='all'):
        """
        Print the chain complex in human-readable format using matplotlib.

        Red arrows: coefficient known to be 1.
        Blue arrows: coefficient unknown.
        Green arrows: zig-zag from cancellations.

        output_file : path (pdf/png/…) or None to display interactively.
        interchanges: list of (g, h) generator index pairs to swap horizontally.
        cancellations: list of (a, b) pairs where d[a→b]=1 to cancel.
        relevant_SpinC: list of SpinC structure indices to show, or 'all'.
        """
        if interchanges is None:
            interchanges = []
        if cancellations is None:
            cancellations = []

        print_equivariantly = self.there_is_action

        if not self.fully_initialized:
            self.generate_domains()
        if relevant_SpinC == 'all':
            relevant_SpinC = self.SpinC

        if self.there_is_action:
            for S in relevant_SpinC:
                if self.image_of_SpinC_structures[S] not in relevant_SpinC:
                    print_equivariantly = False

        replacements = [None] * len(self.generators)
        for g in self.generators:
            if self.SpinC[g] in relevant_SpinC:
                replacements[g] = repr(g)
            else:
                replacements[g] = ''

        arrows = dict()
        for g in self.generators:
            for h in self.generators:
                if (self.can_contribute(g, h)
                        and self.SpinC[g] in relevant_SpinC
                        and self.SpinC[h] in relevant_SpinC):
                    if self.does_contribute(g, h):
                        arrows[(g, h)] = 'red'
                    else:
                        arrows[(g, h)] = 'blue'

        # Cancellations
        for (a, b) in cancellations:
            if self.abs_gr[a] != self.abs_gr[b] + 1:
                raise Exception("Cancelling arrow not in correct grading")
            if arrows.get((a, b)) != 'red':
                print("WARNING: Trying to cancel an arrow where the program "
                      "doesn't know if the coefficient is 1")
                raise Exception(
                    "Not clear if the cancelling arrow is coefficient 1")

            maps_to_a = [foo for foo in self.generators if (foo, a) in arrows]
            maps_to_b = [foo for foo in self.generators
                         if (foo, b) in arrows and foo != a]
            maps_from_a = [foo for foo in self.generators
                           if (a, foo) in arrows and foo != b]
            maps_from_b = [foo for foo in self.generators if (b, foo) in arrows]

            for g in maps_to_b:
                if arrows[(g, b)] == 'red':
                    replacements[g] += '+' + replacements[a]
                else:
                    replacements[g] += '+c(' + replacements[a] + ')'
                for h in maps_from_a:
                    if arrows[(g, b)] == 'red' and arrows[(a, h)] == 'red':
                        if (g, h) in arrows:
                            if arrows[(g, h)] == 'red':
                                del arrows[(g, h)]
                            else:
                                arrows[(g, h)] = 'green'
                        else:
                            arrows[(g, h)] = 'red'
                    else:
                        arrows[(g, h)] = 'green'

            for g in maps_to_a:
                del arrows[(g, a)]
            for g in maps_to_b:
                del arrows[(g, b)]
            for h in maps_from_a:
                del arrows[(a, h)]
            for h in maps_from_b:
                del arrows[(b, h)]

            replacements[a] = ''
            replacements[b] = ''
            del arrows[(a, b)]

        if not self.grading_spread:
            return

        width = max(len(self.grading_spread[gr])
                    for gr in self.grading_spread) + 1
        x_coordinates = [None] * len(self.generators)

        # Assign initial x-positions
        for gr in self.grading_spread:
            ll = [g for g in self.grading_spread[gr]
                  if self.SpinC[g] in relevant_SpinC]
            if not ll:
                continue
            if print_equivariantly:
                fixed = [g for g in ll
                         if self.image_of_generators[g] == g]
                non_fixed = [g for g in ll
                             if self.image_of_generators[g] != g]
                distance = width / (len(non_fixed) + 2)
                for g in fixed:
                    x_coordinates[g] = 0.0
                non_fixed_reps = []
                for g in non_fixed:
                    if self.image_of_generators[g] not in non_fixed_reps:
                        non_fixed_reps.append(g)
                for ind_g, g in enumerate(non_fixed_reps):
                    x_coordinates[g] = (ind_g + 1) * distance
                    x_coordinates[self.image_of_generators[g]] = \
                        -(ind_g + 1) * distance
            else:
                distance = width / (len(ll) + 1)
                for ind_g, g in enumerate(ll):
                    x_coordinates[g] = -width / 2 + (ind_g + 1) * distance

        # Interchanges
        for (a, b) in interchanges:
            if self.abs_gr[a] != self.abs_gr[b]:
                raise Exception("Trying to interchange stuff in different gradings")
            x_coordinates[a], x_coordinates[b] = \
                x_coordinates[b], x_coordinates[a]

        # Draw
        y_vals = sorted(self.grading_spread.keys())
        y_span = (y_vals[-1] - y_vals[0]) if len(y_vals) > 1 else 1
        fig_w = max(6.0, width * 1.2 + 3.0)
        fig_h = max(4.0, y_span * 0.9 + 2.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        for gr in self.grading_spread:
            ax.plot([-width/2, width/2], [gr, gr],
                    alpha=0.1, color='gray', linewidth=1)
            ax.text(width/2 + 0.4, gr, repr(gr) + '+C',
                    color='blue', ha='left', va='center', fontsize=9)

        for g in self.generators:
            if replacements[g] != '' and x_coordinates[g] is not None:
                ax.text(x_coordinates[g], self.abs_gr[g], replacements[g],
                        color='black', ha='center', va='center', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.15',
                                  facecolor='white', edgecolor='none',
                                  alpha=0.8))
                for h in self.generators:
                    if (g, h) in arrows and x_coordinates[h] is not None:
                        ax.plot([x_coordinates[g], x_coordinates[h]],
                                [self.abs_gr[g], self.abs_gr[h]],
                                alpha=0.8, color=arrows[(g, h)], linewidth=1)

        ax.axis('off')
        plt.tight_layout()
        if output_file is None:
            plt.show()
        else:
            fig.savefig(output_file, bbox_inches='tight', dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Standalone function
# ---------------------------------------------------------------------------

def branched_double(H, num_pointed_regions):
    """
    Input a HeegaardDiagram H with 2 basepoints (last two regions).
    Return the double branched cover as a HeegaardDiagram with Z/2-action.

    num_pointed_regions: 1 → single basepoint on preimage of last region;
                         2 → basepoints on preimages of both pointed regions.
    (Requires the knot to be Z/2-nullhomologous.)
    """
    if len(H.regions) - len(H.regions_un) != 2:
        raise Exception("Need two pointed regions")
    if num_pointed_regions not in [1, 2]:
        raise Exception("Can only return 1 or 2 basepoints")

    # Find a path between the two basepoint regions through alpha arcs.
    shortest_path = H.region_graph_alpha.shortest_path(
        H.regions[-2], H.regions[-1])
    cut_edge = [
        next(e[2] for e in H.region_graph_alpha.edges(labels=True)
             if sorted(list(e[:2])) == sorted(shortest_path[steps:steps+2]))
        for steps in range(len(shortest_path) - 1)]

    # Ensure the cut edge meets each alpha circle an even number of times.
    cut_edge_vector = np.array(
        [len([e for e in cut_edge if e[0] == a]) for a in H.alphas],
        dtype=np.uint8) % 2
    incidence_matrix = GF2Matrix(
        len(H.alphas), len(H.betas), H.intersection_matrix)
    try:
        extra_betas = incidence_matrix.solve_right(cut_edge_vector)
    except ValueError:
        raise Exception(
            "Need the knot to be null-homologous mod 2 for the program "
            "to compute double branched cover.")

    for b in H.betas:
        if extra_betas[b] % 2 == 1:
            for p in H.intersections_on_betas[b]:
                [a, n] = H.intersection_incidence[p][0::2]
                if n == 1:
                    cut_edge.append((a, p))
                else:
                    ind_p = H.intersections_on_alphas[a].index(p)
                    cut_edge.append(
                        (a, H.intersections_on_alphas[a][ind_p-1]))

    # Build the double cover.  Most objects have a 0-lift and a 1-lift.
    new_intersections = ([(p, 0) for p in H.intersections]
                         + [(p, 1) for p in H.intersections])
    new_regions = ([(R, 0) for R in H.regions_un]
                   + [(R, 1) for R in H.regions_un]
                   + [(R, -1) for R in H.regions[-2:]])

    new_image_of_intersections = (
        [p + len(H.intersections) for p in H.intersections]
        + [p for p in H.intersections])

    new_boundary_intersections = []
    for (R, i) in new_regions:
        new_boundary_intersections.append([])
        curr_sheet = abs(i)
        for ind_p in range(0, len(H.boundary_intersections[R]), 2):
            [curr_p, next_p] = H.boundary_intersections[R][ind_p:ind_p+2]
            curr_alpha = H.intersection_incidence[curr_p][0]
            new_boundary_intersections[-1].append(
                new_intersections.index((curr_p, curr_sheet)))

            curr_p_ind = H.intersections_on_alphas[curr_alpha].index(curr_p)
            if H.intersections_on_alphas[curr_alpha][curr_p_ind-1] == next_p:
                first_p = next_p
            else:
                first_p = curr_p
            for foo in cut_edge:
                if (curr_alpha, first_p) == foo:
                    curr_sheet = 1 - curr_sheet

            new_boundary_intersections[-1].append(
                new_intersections.index((next_p, curr_sheet)))

        if i == -1:
            new_boundary_intersections[-1] += [
                new_image_of_intersections[p]
                for p in new_boundary_intersections[-1]]

    return HeegaardDiagram(
        new_boundary_intersections,
        num_pointed_regions,
        new_image_of_intersections)
