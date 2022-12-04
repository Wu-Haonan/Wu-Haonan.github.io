---
layout: article
title: Statistical and machine learning methods for spatially resolved transcriptomics data analysis
tags: papers

article_header:
  type: cover
  image: 
    src: /post_image/Spatial_tran/cover.png width="60%"
---

The spatial transcriptome is a novel and cutting-edge field and can identify single-celler transcriptome and corresponding locations. This paper reviews the current statistical and machine-learning methods in Spatial transcriptome. And it's also my first paper read in this field. First, let us introduce the wet-lab technologies in this area to show what spatial transcriptome is.

# Intro to Spatial Transcriptome Technologies

Why do we need ST? Single-cell RNA-seq can provide a perspective on the gene expression level at a cell level. But, when we prepared the sequencing for a single cell, we lost all the information of its location in vivo. The cell position is critical to identify the cell type, state, and function for us. Moreover, many cell communications are based on surface-bound protein receptor-ligand pairs, meaning many signals are passed via neighboring cells or in the same tissue. However, we should give scRNA-seq an objective evaluation that this technology has permitted <b>unbiased</b>(i.e., characterizing the expression of every gene in the genome), <b>genome-scale assessments</b> of cellular identity, heterogeneity, and dynamic change for thousands to hundreds of thousands of cells [^1].

Broadly, there are two ways to complete the goal of detecting transcriptome while preserving location information. Actually, the development of Spatial Transcriptome technologies has lasted for decades, and there are many technologies to pursue this aim. If you are interested, move to [this review](https://www.nature.com/articles/s41592-022-01409-2) or [this blog](https://pachterlab.github.io/LP_2021/index.html) to learn more details about the history of ST.

<p align="center">
    <img src="/post_image/Spatial_tran/History_of_ST.PNG" width="100%">
</p>

__Fig.1 Timeline of Spatial Transcriptome Technologies.[^2]__


The first class of methods is <b>image-based</b>, including _in situ hybridization_ (<b>ISH</b>) and <i>in situ sequencing</i> (<b>ISS</b>). ISH methods use gene-specific fluorophore-labeled probes to detect the number of target mRNAs that we have known their sequence, which means we can only identify a few already-known mRNAs in tissue. In terms of ISS, we use fluorophore-labeled bases to amplify transcripts in situ and obtain their sequences. This technology seems perfect for obtaining sequences and positions at the same time. But actually, the read in FISSEQ (a kind of technology of ISS) is around 5-30nt, and only 200 mRNA reads were captured per cell. Compared to it, 40,000 mRNA reads were detected in scRNA-seq.

<p align="center">
    <img src="/post_image/Spatial_tran/ST_tech.PNG" width="100%">
</p>

__Fig.2 Classes of Spatial Transcriptome Technologies.[^1]__

In fact, the most popular and current Spatial Transcriptome Technologies belong to the <b>sequence-based</b> class. The early technology was <b>Laser Capture Microdissection (LCM)</b>. Through the laser (like IR or UV), we can section the tissue into small pieces and detect the RNA-seq for each piece. It still has some disadvantages, such as limited resolution and laborious process. Alright, let's focus on the most advanced and common technology in recent years--the <b>array-based</b> methods.

Tissue was mounted over an array, such that released mRNA was captured locally by spatially-barcoded probes, converted to cDNA, and then sequenced[^1]. Actually, in human body, the diameter of cell is around 5-200 ${ \mu m }$. A disadvantage of these methods is that capture areas do not follow the complex contours of cellular morphology. Hence, cells often straddle multiple capture areas, contributing mRNA to more than one pixel. Even when capture areas are smaller than a single cell (as in HDST), they still lack single-cell resolution, since they capture mRNA merely from a single-cell-sized area. Besides, their spatial resolution and mRNA recovery rates are lower than ISH and ISS-methods. Finally, by relying on a fixed array, transcripts from different cells can be captured at the same spot, meaning that sophisticated analyses are needed to determine what cell types were present at each spot. The process of identifying and quantifying the relative contribution from each cell type in a capture spot is known as <b>deconvolution</b>[^1].


<p align="center">
    <img src="/post_image/Spatial_tran/array_based.PNG" width="100%">
</p>

__Fig.2 Array-based Technologies.[^3]__











[^1]: [An introduction to spatial transcriptomics for biomedical research.](https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-022-01075-1) 
[^2]: [Museum of spatial transcriptomics](https://www.nature.com/articles/s41592-022-01409-2)
[^3]: [Blog: Museum of spatial transcriptomics](https://pachterlab.github.io/LP_2021/index.html)