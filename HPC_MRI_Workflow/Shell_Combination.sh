#!/bin/bash

#################################
#           Variables           #
#################################
 
#Protocol Variables#
start_time=$( date )
echo "Started at $start_time "

#######################################################
###       Start of Script (do not modify below)     ###
#######################################################

#################################
#         Set Variables         #
#################################
topdir= ####
#SUBJ=$(basename "$PWD")
shells=("b2000" "b1000") # both shells to be looped through

((i=SLURM_ARRAY_TASK_ID-1))
#check existence of subjects directory list file
subjects_file=${topdir}/datalist.txt
#subjects_file=/path/to/list/of/specific/subject/directories
if [ ! -e "${subjects_file}" ]; then
  >&2 echo "error: subjects file named datalist.txt does not exist"
  exit 1
fi
IFS=$'\n'; subjects=( $(cat "${subjects_file}") );
SUBJ=${subjects[${i}]}

## New folders within subject (but not the shell)
COMBINED_DIR=combined_shells

#################################
#   Load Modules     #
#################################

module load gcc/10.2.0
module load mrtrix3tissue/5.2.9
module load ants/2.3.1

###NEW STUFF
current_folder=${topdir}/${SUBJ}

# Extract the participant identifier and session number from the folder name
folder_name=$(basename "$current_folder")
participant_id=${folder_name%%m*}
session_num=${folder_name##*m}

# Define the new folder name with BIDS format
SUBJ_SESSION="sub-$participant_id""_ses-$session_num"

### Make subject directory for connectome processing

cd $topdir/Connectome_Processing 

if [ ! -d ${SUBJ_SESSION} ]; then
    mkdir ${SUBJ_SESSION}
fi

cd $topdir/${SUBJ}

#Create results directory 
if [ ! -d "$COMBINED_DIR" ]; then
	mkdir $COMBINED_DIR
fi

######################################################
###        Shell preprocessing            ###
#######################################################
## changes to the correct shell subfolder
for shell in "${shells[@]}"; do

	if [ ! -d "$shell" ]; then
		echo "No $shell directory found. Skipping preprocessing of $shell shell"
	else
		cd $topdir/${SUBJ}/${shell};

        #################################
        #     Combine dwi volumes       #
        #################################

        ##Convert nii to mif
        mrconvert -fslgrad ${shell}.2.5mm.motion_corrected.bvec ${shell}.2.5mm.motion_corrected.bval ${shell}.2.5mm.motion_corrected.nii.gz ${shell}.mif

        ##Extract b0 volumes###
        mrconvert ${shell}.mif -coord 3 0 ${shell}.b0.mif

        ##### copy files to combined directory
        cp ${shell}.mif ${shell}.b0.mif ${topdir}/${SUBJ}/$COMBINED_DIR
        cd ..
    fi
done

###### NEW STUFF #####

cd $topdir/${SUBJ}/$COMBINED_DIR

##Generate transformations##
mrregister -type rigid -rigid b1000_to_b2000_transformation.txt b1000.b0.mif b2000.b0.mif -transformed b1000.b0_in_b2000.mif

##Transform b1000 shell to b2000 space##
mrtransform -linear b1000_to_b2000_transformation.txt b1000.mif b1000_transformed.mif -template b2000.mif

##Merge shells##
mrcat b2000.mif b1000_transformed.mif dwi.mif

##Calculate intersection mask
dwi2mask b1000_transformed.mif b1000_transformed.mask.mif
dwi2mask b2000.mif b2000.mask.mif
mrcalc b1000_transformed.mask.mif b2000.mask.mif -mult dwi.mask.mif

##Convert back to nii and fsl grads##
mrconvert dwi.mif dwi.nii.gz -export_grad_fsl dwi.bvec dwi.bval
mrconvert dwi.mask.mif dwi.mask.nii.gz

##Create a min b0 image for mask checking##
mrconvert b1000_transformed.mif -coord 3 0 b1000_transformed.b0.mif
mrcalc b2000.b0.mif b1000_transformed.b0.mif -min b0.min.mif
mrconvert b0.min.mif b0.nii.gz

###### Bias correction #########

echo " "; echo "Preprocessed DWI to MRtrix files" ; echo $(date) ; echo " "; 

dwibiascorrect ants dwi.mif dwi.bias.mif

echo "Upsampling diffusion data"
mrgrid dwi.bias.mif regrid -vox 1.25 dwi.bias.1.25mm.mif
echo "Creating upsampled mask"
dwi2mask dwi.bias.1.25mm.mif dwi.bias.1.25mm.mask.mif

### copy results to connectome directory 

cp dwi.bias.1.25mm.mif dwi.bias.1.25mm.mask.mif $topdir/${SUBJ_SESSION} 

#######################################################
###                  End of Script                  ###
#######################################################
fin_time=$( date )
echo "Finished at $fin_time "
