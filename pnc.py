#!/usr/bin/env python3
#
#    pnc-py : Copyright (c) 2023-2024 Yrin Eldfjell : GPLv3
#
#    Reference implementation of the PNC (Parallel Neighbourhood Correlation)
#    algorithm described in my M.Sc. thesis.
#    See https://doi.org/10.1371/journal.pcbi.1000063 for a description of
#    the original algorithm and idea of using Neighborhood Correlation to
#    identify homologous proteins.
#
#    EXPERIMENTAL.
#
#    pnc-py is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, version 3 of the License.
#
#    pnc-py is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program, see the LICENSES file.
#    If not, see <https://www.gnu.org/licenses/>.
#
#    Version 0.02
#

import sys, re, itertools, operator
import numpy as np
from math import sqrt


def read_blast_3col_format(file_handles):
    for fh in file_handles:
        for row in fh:
            query_acc, target_acc, bit_score = row.strip().split()
            yield (query_acc, target_acc, float(bit_score))


def neighborhood_correlation(alignments, nc_threshold=0.05):

    def group_alignments_by_target(alignments):
        get_target_accession = operator.itemgetter(1)
        alignments.sort(key=get_target_accession)
        return itertools.groupby(alignments, get_target_accession)

    def all_query_accessions(alignments):
        return set(a[0] for a in alignments)

    def forward_and_reverse_lookup_tables(a_list):
        forward = {el: i for (i, el) in enumerate(a_list)}
        reverse = {i: el for (el, i) in forward.items()}
        return forward, reverse

    # Setup storage and lookup tables.
    query_accs = all_query_accessions(alignments)
    query_fw_lookup, query_rev_lookup = forward_and_reverse_lookup_tables(query_accs)
    n = len(query_accs)
    query_sums_vector = np.zeros(n, dtype="float32")
    query_square_sums_vector = np.zeros(n, dtype="float32")
    cross_terms = {}
    
    # Process all alignments in groups, one reference db entry at a time.
    for target, query_scores in group_alignments_by_target(alignments):
        score_list = []
        for (q_acc, _, score) in query_scores:
            q_id = query_fw_lookup[q_acc]
            score_list.append((q_id, score))
            # Accumulate x_i and x_i**2 for this query seq.
            query_sums_vector[q_id] += score
            query_square_sums_vector[q_id] += score**2
        # Calculate cross-terms for all (x_i, y_i) aligning to the current ref-db seq.
        for ((q1_id, x), (q2_id, y)) in itertools.combinations(score_list, 2):
            key = (q1_id, q2_id)
            if key in cross_terms:
                cross_terms[key] += x * y
            else:
                cross_terms[key] = x * y

    # Compute the Pearson correlation coefficient for all pairs we found.
    for (x, y), sum_xy in cross_terms.items():
        sum_xx, sum_yy = query_square_sums_vector[x], query_square_sums_vector[y]
        sum_x,  sum_y  = query_sums_vector[x], query_sums_vector[y]
        avg_x,  avg_y  = sum_x / n, sum_y / n
        r_xy_numerator = sum_xy - n*avg_x*avg_y
        root_term_xx = sum_xx - n*avg_x**2
        root_term_yy = sum_yy - n*avg_y**2
        if root_term_xx < 0:
            print("root_term_xx < 0", root_term_xx, file=sys.stderr)
            root_term_xx = 0
        if root_term_yy < 0:
            print("root_term_yy < 0", root_term_yy, file=sys.stderr)
            root_term_yy = 0
        r_xy_denominator = sqrt(root_term_xx) * sqrt(root_term_yy)
        r_xy = r_xy_numerator / r_xy_denominator
        if r_xy >= nc_threshold:
            yield query_rev_lookup[x], query_rev_lookup[y], r_xy


def main():
    alignments = list(read_blast_3col_format([open(f, 'r') for f in sys.argv[1:]]))
    for query, target, nc_score in neighborhood_correlation(alignments):
        print("{} {} {:.3}".format(query, target, nc_score))


main()
