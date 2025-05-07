#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 11:21:54 2025

@author: sidra
"""

# Spatial Transcriptomics
## 10X Genomics Visium data

# Import Python libraries
import os
import anndata
import requests
import tarfile
import numpy as np
import scanpy as sc
import pandas as pd
import squidpy as sq
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

adata = sc.datasets.visium_sge(sample_id="V1_Human_Lymph_Node")
adata.var_names_make_unique()

adata # AnnData object

# Add annotation for mitochondrial genes
adata.var["mt"] = adata.var_names.str.startswith("MT-")

# Calculate quality control metrics
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

# filter out spatial spots with very few detected genes
sc.pp.filter_cells(adata, min_genes=200)

# NORMALIZATION: depends on how your data looks
# 10X Genomics Visium data is a gene-by-spot matrix 
# where each spot are transcripts 
# from a multi-cell region of 
# a tissue, making it more similar 
# to bulk RNA-seq samples 
# that have spatial information

# visualize total counts per spot:

sq.pl.spatial_scatter(
    adata,
    color="total_counts",
    size=1.5
)

# In this case apply standard scRNA-seq normalization
# Normalize to 10,000 (1e4) total counts per spot
sc.pp.normalize_total(adata, target_sum=1e4)

# Log transform the data
sc.pp.log1p(adata)

# select highly variable genes for downstream analyses
sc.pp.highly_variable_genes(adata)


# dimension reduction and data visualization

# Run PCA
## PCA Princpal Component Analysis
## data is linearly transformed onto a 
## new coordinate system where the
## directions - principal components -  
## capture the largest variation 
sc.pp.pca(adata, n_comps=50)

# Calculate k-nearest neighbors
## KNN a non-parametric, supervised learning classifier
## which uses proximity to make classifications or 
## predictions about the grouping of an individual data point.
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

# Leiden clustering
##  community detection algorithm
## that optimizes modularity 
## in extracting communities from networks
## modularity: measure of the structure of networks 
## or graphs which measures the strength of division
## of a network into modules (also called groups, clusters, communities)
sc.tl.leiden(adata, resolution=0.8)

# Computing UMAP
## UMAP Uniform Manifold Approximation and Projection 
## UMAP is a dimension reduction technique that can be 
## used for visualisation 
sc.tl.umap(adata)

# Visualizing marker genes
sc.pl.umap(adata, color=['leiden', 'CD19', 'MS4A1', 'FCER2'], cmap = 'Blues')
sc.pl.umap(adata, color=['leiden', 'CD19', 'FAM138A', 'OR4F5'], cmap = 'Blues')
adata.var
adata.obs

# histology image of tissue section provided by Visium
img = adata.uns['spatial']['V1_Human_Lymph_Node']['images']['hires']
print(f"Shape: {img.shape}, Type: {type(img)}, Dtype: {img.dtype}")

# First top left pixels
img[:3, :3, :]

sq.pl.spatial_scatter(adata)
sq.pl.spatial_scatter(
    adata,
    color="total_counts",
    size=1.5
)

# image container
img = sq.im.ImageContainer.from_adata(adata)

# adding "image_features" to data.obsm
sq.im.calculate_image_features(
    adata,
    img,
    features="summary",
    key_added="image_features"
)
adata

# Transfer to adata.obs for graphing
adata.obs["summary_ch-0_mean"] = adata.obsm["image_features"]["summary_ch-0_mean"]
sq.pl.spatial_scatter(
    adata,
    color="summary_ch-0_mean",
    size=1.5
)

# clustering on image features
sc.pp.neighbors(adata, use_rep="image_features")
sc.tl.leiden(adata, key_added="image_features_clusters", resolution = 0.1)

#adata.uns.pop('image_features_clusters_colors')
sq.pl.spatial_scatter(
    adata,
    color="image_features_clusters",
    size=1.5
)

# Extract spatial coordinates
coords = adata.obsm["spatial"]
print(f"Shape: {coords.shape}, Type: {type(coords)}, Dtype: {coords.dtype}")

# First few coordinates
coords[:3, :]

# spatial neighborhood graph
sq.gr.spatial_neighbors(adata, coord_type="grid") # graph is stored in adata.obsp["spatial_connectivities"]
adata.obsp["spatial_connectivities"].shape # outputs that it is a 2D array/matrix in rows x columns format

# visualize graph to see how spots are linked
sq.pl.spatial_scatter(adata, connectivity_key="spatial_connectivities")

# visualize expression of specific genes directly on histology image of tissue
# to see how specific genes are spatially spread out and how it relates to tissue structure
# CD3E: CD3 epsilon subunit of the T-cell receptor complex
# crucial protein involved in T-cell development and function
sq.pl.spatial_scatter(adata, color = 'CD3E')
adata.obs.sort_values(by="n_genes", ascending=False)

# IGKC: human gene that encodes constant domain of the kappa light chain of Abs
sq.pl.spatial_scatter(adata, color = 'IGKC')

# EEF1A1: eukaryotic translation elongation factor 1 alpha 1
# protein with crucial role in protein synthesis
sq.pl.spatial_scatter(adata, color = 'EEF1A1')
adata.var.sort_values(by="log1p_mean_counts", ascending=False)

# spatial map of gene expression clusters through Leiden
sq.pl.spatial_scatter(adata, color = 'leiden')

# look at marker genes of specific clusters
sc.tl.rank_genes_groups(adata, 'leiden')
sc.pl.rank_genes_groups(adata, n_genes=25, groups = ['2'])
sc.pl.rank_genes_groups(adata, n_genes=25, groups = ['9']) # interferon-induced genes

# visualize OAS1 gene on tissue to see its spatial location and gene expression activity 
# OAS1 I s known interferon-stimulated gene
sq.pl.spatial_scatter(adata, color = ['OAS1'])

# identify genes with spatial structure - genes with expression levels non-randomly 
# distributed across tissue with Moran's I = spatial autocorrelation measure where ↑ Moran's I
# is expression is non-random
sq.gr.spatial_autocorr(adata, mode="moran")

# Top spatially structured genes
adata.uns["moranI"]["I"].sort_values(ascending = False)[0:15]

# visualize some of the top spatially structured genes
sq.pl.spatial_scatter(adata, color = ['IGKC','IGHG2','IGHG4'])

# take principal components (PCs) from PCA of gene count matrix 
# and concatenate them with spatial coordinates of each spot
# Extract PCA and spatial coordinates
X_pca = adata.obsm["X_pca"][:, :10]  # use first 10 PCs
X_spatial = adata.obsm["spatial"]

# Scale values as the spatial coordinates have much larger magnitudes than the PCs
scaler_pca = StandardScaler()
X_pca_scaled = scaler_pca.fit_transform(X_pca)

scaler_spatial = StandardScaler()
X_spatial_scaled = scaler_spatial.fit_transform(X_spatial)

# Combine scaled spatial and scaled pea data into one matrix
X_combined = np.concatenate([X_pca_scaled, X_spatial_scaled], axis=1)
adata.obsm["X_expr_spatial_scaled"] = X_combined

# Build neighbor graph of combined scaled data and cluster using combined features with Leiden
sc.pp.neighbors(adata, use_rep="X_expr_spatial_scaled")
sc.tl.leiden(adata, key_added="leiden_expr_spatial_scaled", resolution = 0.5)

# Visualize the combined data coloring the scaled combined Leiden data
sq.pl.spatial_scatter(adata, color="leiden_expr_spatial_scaled", size=1.5)

adata
