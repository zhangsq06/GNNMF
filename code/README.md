# GraphPred

 GraphPred is a tool based on graph neural network (GNN) and coexisting probability for predicting transcription factor binding sites (TFBSs) and finding multiple motifs from ATAC-seq data.

![1649817641616.png](image/readme/1649817641616.png)

## Dependencies

Python3.6,Biopython-1.70, tensorflow-1.14.0, [HINT](https://www.regulatory-genomics.org/hint/introduction/) and [TOBIAS](https://github.com/loosolab/TOBIAS)

## Data

Downloading the bed (ENCFF848NZM.bed) file and bam (ENCFF405HAW.bam) file of GSE114204 from [the ENCODE project](https://www.encodeproject.org/), and [hg38.fa](https://hgdownload.cse.ucsc.edu/goldenpath/hg38/bigZips/) .

## Step1 Detecting foorprints via the detect_footprints.py, and return foorprints to the dir './final_hint_tobias/'

CMD: python detect_footprints.py --bed bed_file --path current_dir
This CMD need the bed file of hg38.fa *.bed  and *.bam as inputs and output the foorprints
Arguments --bed (the prefix name of your file. For example, if your input files are GSE114204.bed and GSE114204.bam, you need to input GSE114204) --path (current dir which contains hg38.fa *.bed  and *.bam, and a folder named final_hint_tobias will be created under this dir to store footprints)

## Step2 Obtaining training sequences and testing sequences,and return results to the dir './data/encode_101'

CMD python getchrom_index.py --bed bed_name --path dir --peak_flank 50
This CMD need the bed file of footprints as inputs, and output a dataset the are composed of training sequences and testing sequences
Arguments --bed (the bed file of footprints , such as GSE114204) --path  (the dir  that includes the footprints and hg 38.fa) --peak_flank (length of sequecnes in fasta file))

## Step4 Generating adjacency matrixs

CMD python cal_mat.py --dataset sequence_data --k 5
This CMD generate the adjacency matrix that contains similarity matrix, coexsiting matrix and inclusive matrix.

Arguments --dataset (the name of sequence data , such as GSE114204_encode) --k (the length of k-mer)

## Step5 Training GraphPred model

cd ./GraphPred
CMD python train.py --dataset name
This CMD trains models on the  dataset
Arguments --dataset (the name of your data , such as GSE114204_encode)

## Step6 Predicting TFBSs from ATAC-seq data

CMD python locate_tfbs.py --dataset name
this CMD need the dataset as input and ouput the predicted tfbs seq
Arguments --dataset (the name of your data , such as GSE114204_encode) --k (the length of k-mer)

# Step7 Merging TFBSs

CMD python merge_tfbs.py --tfbs name
This CMD need the tfbs seq as input and ouput the mergered tfbs
Arguments --tfbs (the name of the tfbs seq , such as GSE114204_encode.fa)
