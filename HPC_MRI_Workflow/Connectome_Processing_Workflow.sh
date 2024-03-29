#!/bin/bash
###################################################
##                 Script info                   ## 
###################################################

#Script to generate connectome matrices from Diffusion MRI data Using MRtrix3
#and FSL in preterm born children


#################################
#           Variables           #
#################################

#Protocol Variables#
start_time=$( date )
echo "Started at $start_time"
topdir=/bulk/bray_bulk/David_VPT_ARC

#######################################################
###       Start of Script (do not modify below)     ###
#######################################################

i=$(($SLURM_ARRAY_TASK_ID - 1))

#check existence of subjects directory list file
subjects_file=${topdir}/Testing/subject_lists/VSD_subjects_3.txt

#subjects_file=/path/to/list/of/specific/subject/directories
if [ ! -e "${subjects_file}" ]; then
  >&2 echo "error: subjects_file does not exist"
  exit 1
fi

# read the subjects from the subjects file
IFS=$'\n'; subjects=( $(cat "${subjects_file}") );

subject=${subjects[${i}]}


#FSL Environment#
module load fsl/6.0.0
module load matlab/r2020a
module load gcc/10.2.0
module load ants/2.3.1
module load mrtrix3tissue/5.2.9

####### Set folders##########

#Navigate to working directory

SPM_DIR=${topdir}/software/matlab/spm12
SCRIPTS_DIR=${topdir}/scripts

cd ${topdir}/Testing/VSD_subjects/${subject}


####################################### 
### Modifying the T1 structural image ###
#######################################
echo " ": echo ${subject}"T1 Betting started at"; echo $(date) ; echo " ";

bet ${subject}_T1w.nii.gz brain.nii.gz -B -f 0.1

mrconvert ${subject}_T1w.nii.gz T1_raw.mif

#######################################
###         DWI Preprocessing       ###
#######################################

#Check the time#
echo " "; echo ${subject}"DWI Processing started at" ; echo $(date) ; echo " "; 

#Preprocess DWI - already done as part of VSD_attn_noddi_2019
dwi2response dhollander dwi.bias.1.25mm.mif response_wm.txt response_gm.txt response_csf.txt

#Estimate WM FODs
dwi2fod msmt_csd dwi.bias.1.25mm.mif response_wm.txt wmfod.mif response_gm.txt gm.mif response_csf.txt csf.mif -mask dwi.bias.1.25mm.mask.mif

#Bias field and intensity normalisation
mtnormalise wmfod.mif wmfod_norm.mif gm.mif gm_norm.mif csf.mif csf_norm.mif -mask dwi.bias.1.25mm.mask.mif

#Check the time#
echo " "; echo ${subject}"DWI Processing finished at" ; echo $(date) ; echo " "; 

#######################################
###         DWI Preprocessing       ###
#######################################

#Check the time#
echo " "; echo ${subject} "DWI Processing started at" ; echo $(date) ; echo " "; 

#Preprocess DWI - already done as part of VSD_attn_noddi_2019
dwi2response dhollander dwi.bias.1.25mm.mif response_wm.txt response_gm.txt response_csf.txt

#Estimate WM FODs
dwi2fod msmt_csd dwi.bias.1.25mm.mif response_wm.txt wmfod.mif response_gm.txt gm.mif response_csf.txt csf.mif -mask dwi.bias.1.25mm.mask.mif

#Bias field and intensity normalisation
mtnormalise wmfod.mif wmfod_norm.mif gm.mif gm_norm.mif csf.mif csf_norm.mif -mask dwi.bias.1.25mm.mask.mif

#Check the time#
echo " "; echo ${subject}"DWI Processing finished at" ; echo $(date) ; echo " "; 

#### Betting the T1 structural image#####
echo " ": echo ${subject}"T1 Betting started at"; echo $(date) ; echo " ";

#Extract mean b0 volume from upsampled DWI for T1 registration
dwiextract dwi.bias.1.25mm.mif - -bzero | mrmath - mean mean_b0.nii.gz -axis 3
mrcalc mean_b0.nii.gz dwi.bias.1.25mm.mask.mif -mult mean_b0_brain.nii.gz

#################################
#REGISTER dwi images (mean_bo) to structural (brain.nii.gz)
#################################
mrconvert brain.nii.gz brain.mif ## brain is T1 non coreg

flirt -ref brain.nii.gz -in mean_b0_brain.nii.gz -omat diff2struct_fsl.mat -dof 6 #register diffusion against BET T1; generates B0_brain_to_T1

transformconvert diff2struct_fsl.mat mean_b0_brain.nii.gz brain.mif flirt_import diff2struct_mrtrix.txt #generates B0_brain_to_T1_inverse

mrtransform brain.mif -linear diff2struct_mrtrix.txt -inverse T1_coreg.mif  #get BET T1 registered to diffusion, equivalent to brain_difOants_b0brain.nii.gz

################################
### Tractography  ###
################################

#Generate 5tt image
5ttgen fsl T1_raw.mif 5tt_no_coreg.mif -nthreads 0 

## align 5tt image with dwi
mrtransform 5tt_no_coreg.mif -linear diff2struct_mrtrix.txt -inverse 5tt_coreg.mif

tckgen wmfod_norm.mif tracts_20mil_ACT.tck -act 5tt_coreg.mif -backtrack -crop_at_gmwmi -select 20000000 -seed_dynamic wmfod_norm.mif
tcksift2 tracts_20mil_ACT.tck wmfod_norm.mif tcksift2_weights_ACT_20mil.csv -act 5tt_coreg.mif ## output is weights file, 5tt is segmented anatomical image used for mask
tckedit tracts_20mil_ACT.tck -num 100k tracks_100k_sift.tck
tckedit tracts_20mil_ACT.tck -num 10k tracks_10k_sift.tck

##################################
### Transform images to shared space
##################################

#Warp T1 subject to MNI space for use in atlas
flirt -ref ${SCRIPTS_DIR}/MNI152_T1_2mm_brain.nii.gz -in brain.nii.gz -omat my_affine_transf.mat
fnirt --in=${subject}_T1w.nii.gz --aff=my_affine_transf.mat --cout=my_nonlinear_transf --config=T1_2_MNI152_2mm
applywarp --ref=${SCRIPTS_DIR}/Template_T1.nii.gz --in=${subject}_T1w.nii.gz --warp=my_nonlinear_transf --out=T1_in_template_space

#MNI to T1
invwarp -r ${subject}_T1w.nii.gz -w my_nonlinear_transf.nii.gz -o MNI_to_T1_warp
applywarp -r ${subject}_T1w.nii.gz -i ${SCRIPTS_DIR}/Template_T1.nii.gz -w MNI_to_T1_warp -o Template_T1_in_native_space --interp=nn


#Warp Atlases to T1 space
applywarp -r ${subject}_T1w.nii.gz -i ${SCRIPTS_DIR}/Schaefer_200Parcels_7Networks_Tian_Subcortex_S2_MNI152NLin6Asym_1mm.nii.gz -w MNI_to_T1_warp -o Atlas_native_200.nii.gz --interp=nn
applywarp -r ${subject}_T1w.nii.gz -i ${SCRIPTS_DIR}/Schaefer_400Parcels_7Networks_Tian_Subcortex_S3_MNI152NLin6Asym_1mm.nii.gz -w MNI_to_T1_warp -o Atlas_native_400.nii.gz --interp=nn

# Warp Atlases in T1 space to DWI space
mrconvert Atlas_native_200.nii.gz Atlas_native_200.mif
mrconvert Atlas_native_400.nii.gz Atlas_native_400.mif
mrtransform Atlas_native_200.mif -linear diff2struct_mrtrix.txt -inverse Atlas_b0brain_200.mif
mrtransform Atlas_native_400.mif -linear diff2struct_mrtrix.txt -inverse Atlas_b0brain_400.mif
mrconvert Atlas_b0brain_200.mif Atlas_b0brain_200.nii.gz
mrconvert Atlas_b0brain_400.mif Atlas_b0brain_400.nii.gz

####################################
###   Generate Connectome    200    ###
####################################

#Check the time#
echo " "; echo ${subject}"Generating 200 Connectome started at" ; echo $(date) ; echo " "; 

#Make OUTPUT_200 directory
OUTPUT_200=Schaefer_200_atlas
mkdir $OUTPUT_200

#Generate a connectome of UNweighted streamline count
tck2connectome -force tracts_20mil_ACT.tck Atlas_b0brain_200.nii.gz $OUTPUT_200/connectome_tckcount.csv -zero_diagonal -symmetric

#Generate a connectome of SIFT2 streamline count
tck2connectome -force tracts_20mil_ACT.tck Atlas_b0brain_200.nii.gz $OUTPUT_200/connectome_SIFT2_tckcount.csv -zero_diagonal -symmetric -tck_weights_in tcksift2_weights_ACT_20mil.csv

#Generate a connectome weighted by Inverse of Node Volumes
tck2connectome -force tracts_20mil_ACT.tck Atlas_b0brain_200.nii.gz $OUTPUT_200/connectome_InvNodeVol.csv -zero_diagonal -symmetric -scale_invnodevol -stat_edge mean


###################################
###   Generate Connectome   400     ###
####################################

#Check the time#
echo " "; echo ${subject}"Generating 400 Connectome started at" ; echo $(date) ; echo " "; 

#Make OUTPUT_400 directory
OUTPUT_400=Schaefer_400_atlas
mkdir $OUTPUT_400

#Generate a connectome of UNweighted streamline count
tck2connectome -force tracts_20mil_ACT.tck Atlas_b0brain_400.nii.gz $OUTPUT_400/connectome_tckcount.csv -zero_diagonal -symmetric

#Generate a connectome of SIFT2 streamline count
tck2connectome -force tracts_20mil_ACT.tck Atlas_b0brain_400.nii.gz $OUTPUT_400/connectome_SIFT2_tckcount.csv -zero_diagonal -symmetric -tck_weights_in tcksift2_weights_ACT_20mil.csv

#Generate a connectome weighted by Inverse of Node Volumes
tck2connectome -force tracts_20mil_ACT.tck Atlas_b0brain_400.nii.gz $OUTPUT_400/connectome_InvNodeVol.csv -zero_diagonal -symmetric -scale_invnodevol -stat_edge mean

