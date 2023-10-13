import data_process
import model
from Parameter_settings import set_param
import computer_metapath

metapath_found = computer_metapath.metapath_seek()

adjM, type_mask = metapath_found.build_adjacency_matrix()
print("Adjacency matrix established!")

expected_metapath = [
			[(0, 1, 0), (0, 1, 2, 1, 0)],
			[(1, 0, 1), (1, 0, 2, 0, 1)],
			[(2, 1, 2), (2, 1, 0, 1, 2)]
		]

metapath_found.get_metapath_neighbor_pairs(adjM, type_mask, expected_metapath[0], 0)
print("Author type nodes have completed neighbor collection based on meta paths!")
metapath_found.get_metapath_neighbor_pairs(adjM, type_mask, expected_metapath[1], 1)
print("Paper type nodes have completed neighbor collection based on meta paths!")
metapath_found.get_metapath_neighbor_pairs(adjM, type_mask, expected_metapath[2], 2)
print("Venue type nodes have completed neighbor collection based on meta paths!")


