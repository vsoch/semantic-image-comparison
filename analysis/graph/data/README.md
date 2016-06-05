# Data Folder Includes:

- [concept_kindof_df.tsv](concept_kindof_df.tsv): a data frame of concepts by concepts with all cognitive concepts implicated in this analysis. The relationship is represented as (row)--kindof-->(column)

- [concept_partof_df.tsv](concept_partof_df.tsv): a data frame of concepts by concepts with all cognitive concepts implicated in this analysis. The relationship is represented as (row)--partof-->(column)

- [concept_both_df.tsv](concept_both_df.tsv): both combined.

- [contrast_by_concept_binary_df.tsv](contrast_by_concept_binary_df.tsv): A data frame of contrasts by concepts, where a 1 indicates the contrast is tagged by the concept.

- [contrast_defined_images_filtered.tsv](contrast_defined_images_filtered.tsv): The original image meta data file (produced by pyneurovault) with images filtered down and included in this analysis.

- [graphs_networkx.pkl](graphs_networkx.pkl): a pickled object (dict) of networkx graphs for each of the kind_of and part_of relationships, and both compiled.
