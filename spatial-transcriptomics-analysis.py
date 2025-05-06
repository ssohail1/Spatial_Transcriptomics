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

# Visualize marker genes through UMAP
sc.pl.umap(adata, color=['leiden', 'CD19', 'MS4A1', 'FCER2'], cmap = 'Reds')




