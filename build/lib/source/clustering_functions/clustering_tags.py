from typing import List, Dict, Union
from ..embedding.embedding_model import Embedding_Model
from ..document_functions.text_heuristics import word_normalization
from sklearn.cluster import DBSCAN
import numpy as np


def compute_cluster_tags(
    embedding_model: Embedding_Model,
    tags: List[str],
    **kwargs,
) -> List[Dict[str, Union[str, int]]]:
    """Clusters a list of tags using DBSCAN algorithm.

    Args:
        embedding_model (Embedding_Model): The embedding model used to create tag embeddings.
        tags (List[str]): The list of tags to be clustered.
        **kwargs: Additional keyword arguments for DBSCAN algorithm.

    Returns:
        List[Dict[str, Union[str, int]]]: A list of dictionaries containing the tag, cluster, and frequency.

    """
    # Normalize the tags
    normalized_tags = word_normalization(tags)
    tag2norm = {tag: norm_tag for tag, norm_tag in zip(tags, normalized_tags)}

    # Create the embeddings
    embeddings = embedding_model(normalized_tags)
    unique_tags = {}
    for _tag, _embedding in zip(normalized_tags, embeddings):
        if _tag not in unique_tags:
            unique_tags[_tag] = _embedding

    # Cluster the tags using DBSCAN
    embedding_list = list(unique_tags.values())  # Convert embeddings to a list
    dbscan = DBSCAN(eps=kwargs['eps'], min_samples=kwargs['min_samples'])  # DBscan
    clusters = dbscan.fit_predict(embedding_list)
    tag_clusters = []  # Create a dictionary to store the clusters
    for tag, cluster, embedding in zip(unique_tags.keys(), clusters, embedding_list):
        tag_clusters.append({
            "tag": tag,
            "tag_normalized": tag2norm[tag],
            "cluster": cluster,
            "frequency": normalized_tags.count(tag),
            "embedding": embedding,
        })

    return tag_clusters


def create_tag_edges(
    embedding_model: Embedding_Model,
    tag_lists: List[List[str]],
    **kwargs,
):
    """
    Creates tag edges by replacing tags with the center of normalized tags.

    Args:
        embedding_model (Embedding_Model): The embedding model used for computing tag clusters.
        tag_lists (List[List[str]]): A list of lists containing tags.
        **kwargs: Additional keyword arguments.

    Returns:
        List[List[str]]: A list of lists containing the tag edges.

    """
    # Create tag clusters
    tags = [tag for tag_list in tag_lists for tag in tag_list]
    tag_clusters = compute_cluster_tags(embedding_model, tags, **kwargs)
    tag2norm = {_cluster['tag']: _cluster['tag_normalized'] for _cluster in tag_clusters}

    # Replace tags with the center of normalized tags
    clusters = list(set([_cluster['cluster'] for _cluster in tag_clusters]))
    clusters = list(filter(lambda x: x != -1, clusters))
    if not clusters:
        for _el in tag_lists:
            for i in range(len(_el)):
                _el[i] = tag2norm[_el[i]]
    else:
        tag2cluster_center_tag = {}
        for k in clusters:
            cluster_tags = list(filter(lambda x: x['cluster'] == k, tag_clusters))
            cluster_embeddings = [cluster_tag['embedding'] for cluster_tag in cluster_tags]
            center_embedding = sum(cluster_embeddings) / len(cluster_embeddings)
            closest_tag = min(cluster_tags, key=lambda x: np.linalg.norm(x['embedding'] - center_embedding))
            closest_tag_normalized = closest_tag['tag_normalized']
            for cluster_tag in cluster_tags:
                tag2cluster_center_tag[cluster_tag['tag']] = closest_tag_normalized
        for _el in tag_lists:
            for i in range(len(_el)):
                _el[i] = tag2cluster_center_tag[_el[i]]

    return tag_lists
