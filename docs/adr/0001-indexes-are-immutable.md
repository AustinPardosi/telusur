# Indexes are immutable

An index is built once from a corpus and never changes; there is no add, update or delete, and changing the corpus means a full rebuild. Every on-disk format leans on this: postings store doc ids as gaps from the previous doc, clusters are padded to whole blocks of 32 codes, and centroids are trained once on a sample. Supporting writes would mean segments and merges, or a deleted-docs bitmap on every search path, which is a different project.
