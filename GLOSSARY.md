# telusur

A hybrid search engine: keyword and neural retrieval over one corpus, fused into one ranked list.

## Corpus

**Corpus**:
The ordered set of docs an index is built from.
_Avoid_: dataset (a dataset may yield several corpora), collection

**Doc**:
One item in the corpus: a title, a text, and optional filter fields.
_Avoid_: document (fine in prose, but not in names), record, row, page

**Doc id**:
A doc's position in the corpus, counting from 0. Internal only.
_Avoid_: row, docno, index

**External id**:
The id a doc carries in its source data. The only id a caller ever sees.
_Avoid_: key, doc key, uuid

**Snippet**:
A short, fixed excerpt of a doc's text, the same for every query.
_Avoid_: summary, highlight

**Domain**:
The host of a doc's URL with any leading `www.` removed. Subdomains are distinct domains.
_Avoid_: site, registered domain, eTLD+1

**Crawl date**:
When a doc was fetched by the crawler. Not when it was published.
_Avoid_: date, published date, timestamp

## Search

**Index**:
A read-only structure built once from a corpus. Changing the corpus means building a new index.
_Avoid_: database, store

**Mode**:
How a query retrieves docs: **keyword**, **neural**, or **hybrid** (both, fused).
_Avoid_: dense, sparse, vector, semantic, BM25 (as a mode name)

**Filter**:
A condition on filter fields that a doc must satisfy to be returned. Values for one field are alternatives; different fields must all hold; a doc missing a filtered field never matches.
_Avoid_: facet, where clause

**Selectivity**:
The fraction of the corpus a filter allows.

**Rerank depth**:
How many candidates the first stage of neural retrieval passes on to exact scoring.
_Avoid_: rerank (as a number), candidates

## Neural retrieval

**Full vector**:
A doc's embedding exactly as the model produced it.
_Avoid_: raw vector, embedding (when the distinction matters)

**Rerank vector**:
A shortened, lower-precision copy of a full vector, used for exact scoring.

**Code**:
A doc's 1-bit-per-dimension summary, used for first-stage scoring.
_Avoid_: binary vector, hash, signature

**Cluster**:
One centroid and the docs assigned to it. A query only scores docs in the clusters it probes.
_Avoid_: list, IVF list, bucket, partition

**Block**:
A fixed group of 32 codes scored together.
_Avoid_: batch, chunk

**Group**:
A run of 4 consecutive query dimensions that share one lookup table.
_Avoid_: subvector, chunk

**Rerank ground truth**:
The exact top-k by rerank vectors. Measures what the index loses.

**Full ground truth**:
The exact top-k by full vectors. Measures what the whole design loses, shortening included.

## Keyword retrieval

**Posting list**:
Every doc a term appears in, with how often. One entry is a **posting**.
_Avoid_: list, inverted list

**Tier**:
The part of a posting list whose docs share one term frequency.
_Avoid_: group, bucket, tf group

**Skip interval**:
A fixed run of postings that search can jump over or resume inside.
_Avoid_: block, skip block
