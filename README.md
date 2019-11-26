This software contains a modification of the toolbox MatConvNet (http://www.vlfeat.org/matconvnet/). MatConvNet is a toolbox that implements CNN on Matlab. 

This code implements our solution for automatic melanoma diagnosis, which has been initially submitted to the ISIC 2017 challenge in melanoma diagnosis (https://challenge.kitware.com/#challenge/583f126bcad3a51cc66c8d9a) and then improved for the paper:

I. Gonzalez Diaz, "DermaKNet: Incorporating the knowledge of dermatologists to Convolutional Neural Networks for skin lesion diagnosis," in IEEE Journal of Biomedical and Health Informatics, vol. PP, no. 99, pp. 1-1.
doi: 10.1109/JBHI.2018.2806962 

We have participated in the Part 3: Lesion Classification. In this task, participants are asked to complete two independent binary image classification tasks that involve three unique diagnoses of skin lesions (melanoma, nevus, and seborrheic keratosis). In the first binary classification task, participants are asked to distinguish between (a) melanoma and (b) nevus and seborrheic keratosis. In the second binary classification task, participants are asked to distinguish between (a) seborrheic keratosis and (b) nevus and melanoma.

Definitions:

    Melanoma – malignant skin tumor, derived from melanocytes (melanocytic)
    Nevus – benign skin tumor, derived from melanocytes (melanocytic)
    Seborrheic keratosis – benign skin tumor, derived from keratinocytes (non-melanocytic)

A description of the method is given in the following papers:

Iván Gonzalez Diaz, "DermaKNet: Incorporating the knowledge of dermatologists to Convolutional Neural Networks for skin lesion diagnosis," in IEEE Journal of Biomedical and Health Informatics, vol. PP, no. 99, pp. 1-1. doi: 10.1109/JBHI.2018.2806962

Iván González-Díaz. Incorporating the Knowledge of Dermatologists to Convolutional Neural Networks for the Diagnosis of Skin Lesions. CoRR abs/1703.01976 (2017)

Compiling the code: Compile the code following the instructions of matconvnet. IMPORTANT: Disable double support.

cd matlab
vl_compilenn('EnableGpu',true,'EnableCudnn',true,'EnableDouble', false)


Running the demo: providing a diagnosis for a given image

To run the demo, follow these steps:

	- Compite the MatConvNet toolbox as described in the authors webpage http://www.vlfeat.org/matconvnet/.
	- In demo.m script, modify the corresponding parameters (useGPU,bsize, etc.)
	- Run the demo.m script.

The code operates as follows:

	1.- It accepts as inputs pair of images containing both the dermoscopic lesion and a binary lesion mask separating pixels belonging to the lesion to the surrounding skin.
	2.- It performs a data augmentation process that generates rotated and cropped versions of the original image of the lesion.
	3.- It generates a segmentation of the lesion into a set of 8 dermoscopic features of interest for dermatologists.
	4.- Using the previous information, it computes a diagnosis as a 3 vector containing probabilities of benign, melanoma and seborrheic keratosis. As we have several views of the same lesion due to data augmentation,individual outputs are fused using an average aggregator before applying the softmax. 



Training your own networks


I provide two functions in case you would like to train your own networks:

1.- trainWeakSegmentationNet.m: trains a segmentation CNN into a set of 8 dermoscopic features. It uses the weak annotations about the presence of the demoscopic features:
	- Label 0: the feature is not present.
	- Label 1: the feature is present but local
	- Label 2: the feature is present and global (dominant) in the lesion.
	- Label 3: the feature appears in the borders of the lesion.
        
	The file data/netDefs/segNet.m contains the definition of our current segmentation network and can be used to train your own network in case you have the weak labels.

2.- trainDiagnosisNet.m: trains a diagnosis network. We provide two examples of network structures:
	-data/netDefs/simpleNet.m: a simple network based on resnet-50.
	-data/netDefs/completeNet.m: the structure we used in the ISIC-2017 challenge.


Results
In addition, infolder 'results', I also provide the csv with the test results for two approaches:
1) The official submission to ISIC challenge 2017: out_isic_2017.csv
2) The best result in paper "Iván Gonzalez Diaz, "DermaKNet: Incorporating the knowledge of dermatologists to Convolutional Neural Networks for skin lesion diagnosis," in IEEE Journal of Biomedical and Health Informatics, vol. PP, no. 99, pp. 1-1. doi: 10.1109/JBHI.2018.2806962": out_jbhi2019.csv

For any question regarding the code, send me an e-mail to igonzalez@tsc.uc3m.es







