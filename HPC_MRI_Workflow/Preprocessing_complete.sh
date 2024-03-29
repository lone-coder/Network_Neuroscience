#!/bin/bash
###################################################
##                 Script info                   ##
###################################################

#Script to perform all of diffusion preprocessing; detecting and removing motion corrupt volumes.

#Necessary Input Files
##b1000 .nii.gz, .bvec, .bval
##b2000 .nii.gz, .bvec, .bval

#Output Files (same for both shells)
## b2000 .dropout_free.nii.gz , .bvec  , .bval - multishell data without motion corrupt volumes
## b2000 .dropout_vols.nii.gz - excluded diffusion volumes
## signal_dropout_vols.txt - text file indicating which diffusion volumes were removed from the multishell data
## Pre_Processing_.output , error - files summarizing what the script did and any errors it ran into

#To run script, copy .sh and .slurm into a given subjects data folder on ARC and submit by typing: sbatch DWI_preprocessing_complete.slurm

#######################################################
### User Set Requirments, Variables and Environment ###
#######################################################


#################################
#           Variables           #
#################################

#Processing Variables#
reres=2.5 ######## resampling resolution
dmri_fthreshold=0.3 #fthreshold for fsl masking
num_vols=48 #total number of volumes
num_slices=45 #number of slices in the diffusion data
volume_slice_thr=0.2 #allowable proportion of slices within a volume that can have signal dropout before the volume is removed (0.1 = 10%)
thr_vols=5 #minimum number of motion corrupt volumes that would lead to participant/data set rejection from the study
mporder=15 #corrects for such within-volume (or "slice-to-volume") movement. 
s2v_niter=10 #Specifies the number of iterations to run when estimating the slice-to-vol movement parameters.
shells=("b2000" "b1000") # both shells to be looped through
SUBJ='0876022m12'

#######################################################
###       Start of Script (do not modify below)     ###
#######################################################

echo "Script started at $(date)"

#################################
#    Set Container Environment      #
#################################

#FSL Environment#
export PATH=${FSLDIR}/bin:$PATH
export LD_LIBRARY_PATH=$FSLDIR/lib:$LD_LIBRARY_PATH

####### Set folders##########
topdir=###
#Navigate to working directory
cd $topdir/${SUBJ}


#######################################################
###        Shell preprocessing            ###
#######################################################
## changes to the correct shell subfolder
for shell in "${shells[@]}"; do

	if [ ! -d "$shell" ]; then
		echo "No $shell directory found. Skipping preprocessing of $shell shell"
	else
	
		cd $topdir/${SUBJ}/${shell};

		#Write to preprocessing notes
		notes=${shell}.preprocessing_notes.txt

#################################
#     Run FSL's Eddy_Cuda using symlink to correct version#
#################################


		#Prep files for eddy#
		echo "Prepping files for eddy" >> $notes
		echo $PWD >> $notes

		
		###Resample images to new resolution
		echo "Resampling"  >> $notes
		flirt -in ${shell}.nii.gz -ref ${shell}.nii.gz -out ${shell}.${reres}mm.nii.gz -applyisoxfm $reres
		echo "Resampled dmri images to $reres mm using FSL's flirt" >> $notes

		#Masking
		echo "Masking"  >> $notes
		bet ${shell}.${reres}mm.nii.gz ${shell}.${reres}mm.nii.gz -m -n -f ${dmri_fthreshold} #deleting non-brain tissues
		echo "Masked ${shell} shell with FSL's bet (fthreshold = ${dmri_fthreshold})" >> $notes

		#make an eddy output folder
		EDDY_DIR=eddy.outlier_detection
		mkdir $EDDY_DIR


		#write an index file
		indx=""	
		for ((i=1; i<=$num_vols; i+=1)); do
			indx="$indx 1";
		done
		echo $indx > ${EDDY_DIR}/eddy_cuda_index.txt

		# create acquisition parameters
		printf "0 -1 0 0.170" > ${EDDY_DIR}/acqparams.txt      ######In dennis script

		echo "	fin"


		#Run eddy#
		echo "Running eddy"  >> $notes
		eddy_cuda11.2 --imain=${shell}.${reres}mm.nii.gz --mask=${shell}.${reres}mm_mask.nii.gz --acqp=${EDDY_DIR}/acqparams.txt --bvecs=${shell}.bvec --index=${EDDY_DIR}/eddy_cuda_index.txt --bvals=${shell}.bval --out=${EDDY_DIR}/${shell} --repol --verbose 


		#Check if 1st eddy ran successfully
		check_file="$EDDY_DIR/${shell}.eddy_outlier_free_data.nii.gz"
		if [ ! -e "$check_file" ]; then
			echo "Eddy output not found!" >> $notes
			echo "FSL's eddy failed!" >> $notes
		elif [ -e "$check_file" ]; then

			#conclude step to notes
			echo "Detected signal outliers using FSL's eddy" >> $notes

		fi

		cd ..
	fi
done

#################################
#   Identify corrupt volumes /dmri check#
#################################


#define value for slice-wise volume rejection#
slice_thr_vol="$(echo "$num_slices*$volume_slice_thr" | bc)"
slice_thr_vol=$(printf "%.0f" $slice_thr_vol)

#Navigate to working directory
cd $topdir/${SUBJ}

## folders within subject (but not the shell)
RESULTS_DIR=results
EXCLUDED_DIR=excluded


#Create results directory 
if [ ! -d "$RESULTS_DIR" ]; then
	mkdir $RESULTS_DIR
fi

#Create excluded directory 
if [ ! -d "$EXCLUDED_DIR" ]; then
	mkdir $EXCLUDED_DIR
fi


#set up excluded volume summary file#
summary_excluded_vols=$topdir/${SUBJ}/${RESULTS_DIR}/Signal_Dropout.excluded_vols.txt
if [ ! -e "$summary_excluded_vols" ]; then
	echo "Summary of subject's volumes that were labeled for exclusion due to excessive signal dropout:" >> $summary_excluded_vols
	echo " " >> $summary_excluded_vols
	echo "Subject ID:	Shell:	# of Bad Volumes:	Bad Volume #s:	#of bad slices:">> $summary_excluded_vols
	echo " " >> $summary_excluded_vols
fi

#set up excluded shell summary file#
summary_excluded_shells=$topdir/${SUBJ}/${RESULTS_DIR}/Signal_Dropout.excluded_shells.txt
if [ ! -e "$summary_excluded_shells" ]; then
	echo "Summary of subjects who were excluded due to excessive signal dropout artifacts:" >> $summary_excluded_shells; echo " " >> $summary_excluded_shells
	echo "Subject ID:	Shell:	# of Bad Volumes:	#Number over threshold ($thr_vols volumes)" >> $summary_excluded_shells; echo " " >> $summary_excluded_shells
fi

#set up included result summary file
summary_included_shells=$topdir/${SUBJ}/${RESULTS_DIR}/Signal_Dropout.included_shells.txt
if [ ! -e "$summary_included_shells" ]; then
	echo "Number of volumes removed for subjects who were below the threshold ($thr_vols volumes):" >> $summary_included_shells; echo " " >> $summary_included_shells
	echo "Subject ID:	Shell:	# of Bad Volumes:" >> $summary_included_shells; echo " " >> $summary_included_shells
fi

#################################
#  Set-up Shell preprocessing   #
#################################

for shell in "${shells[@]}"; do

	if [ ! -d "$shell" ]; then
		echo "No $shell directory found. Skipping preprocessing of $shell shell"
	else
	
		cd $topdir/${SUBJ}/${shell};

		#Write to preprocessing notes
		notes=${shell}.preprocessing_notes.txt

#######################################################
###    Step 2: Identify Signal Dropout Volumes      ###
#######################################################

#################################
# Set up input and output files #
#################################

		#Specify input file#
		outliers=eddy.outlier_detection/${shell}.eddy_outlier_report
		#dos2unix -q $outliers

		#Set up text file to record signal dropout volumes#
		touch ${shell}.signal_dropout_vols.txt

		#Create a directory for temporary files#
		TEMP_DIR=temp_dir; mkdir $TEMP_DIR

		#Write a temporary txt file containing a list of volumes with outliers#
		awk '{ print $5 }' $outliers > ${TEMP_DIR}/temp_vols.txt

#################################
#    Check for bad volumes      #
#################################

		#Identify volumes with excessive signal dropout#

		start=0; finish=$num_vols
		for vol in $(eval echo "{$start..$finish}"); do
			vol_occurance=$(grep -c -x $vol ${TEMP_DIR}/temp_vols.txt)

			if (($vol_occurance >= $slice_thr_vol)); then

				echo "$vol" >> ${shell}.signal_dropout_vols.txt
				echo "$vol		$vol_occurance" >> ${TEMP_DIR}/temp_report.txt
				echo "${vol}" >> ${TEMP_DIR}/temp_vol_IDs.txt			
				echo "$vol_occurance" >> ${TEMP_DIR}/temp_vol_occurance.txt

			fi
		done

		#Conclude step to notes
		echo "Identified volumes with excessive signal dropout" >> $notes
		echo "Identified volumes with excessive signal dropout"

		#calculate/format results for summary file#
		num_del_vols=$(wc -l < ${shell}.signal_dropout_vols.txt)
		if (($num_del_vols >= 1)); then
			bad_vol_IDs=$(cat ${TEMP_DIR}/temp_vol_IDs.txt)
			bad_vol_IDs=$(echo $bad_vol_IDs)
			num_bad_slices=$(cat ${TEMP_DIR}/temp_vol_occurance.txt)
			num_bad_slices=$(echo $num_bad_slices)

			echo "The following volumes have more outliers than the threshold (${slice_thr_vol}):" >> $notes
			echo "Volume:	# of bad slices:" >> $notes
			cat ${TEMP_DIR}/temp_report.txt >> $notes
			echo "${SUBJ}		${shell}		${num_del_vols}		${bad_vol_IDs}		${num_bad_slices}">> $summary_excluded_vols
				#echo "${SUBJ}		${shell}		${num_del_vols}		${bad_vol_IDs}		${num_bad_slices}">> Step_5_excluded_vols.txt
		else 
			echo "${shell} shell has no volumes with outliers above the threshold (${slice_thr_vol})" >> $notes
		fi
	
		#remove temporary files directory#
		rm -r $TEMP_DIR

		#######################################################
###    Step 3: Remove Signal Dropout Volumes        ###
#######################################################

#################################
#   Check Inclusion Criteria    #
#################################

		#ensure txt file are in unix format#
		#dos2unix -q ${shell}.signal_dropout_vols.txt

		#get rid of blank lines in txt files#
		sed '/^s*$/d' < ${shell}.signal_dropout_vols.txt > temp.txt; mv temp.txt ${shell}.signal_dropout_vols.txt;

		#Check number of volumes removed due to signal dropout#
		bad_volumes=$( sed -n '=' ${shell}.signal_dropout_vols.txt | wc -l)

		#move shell to excluded directory if total bad volumes is over threshold#
		if (($bad_volumes >=$thr_vols)); then
	
			echo "Total bad volumes in ${shell} (${bad_volumes}) exceeded or is equal to the threshold ($thr_vols)" >> $notes
			echo "${shell} excluded" >> $notes

			dif=$((bad_volumes - thr_vols))
			echo "${SUBJ}		${shell}		${bad_volumes}		${dif}" >> $summary_excluded_shells
				#echo "${SUBJ}		${shell}		${bad_volumes}		${dif}" >> Step_5_excluded_shells.txt
			#move shell data if excluded#
			NEW_DIR=${EXCLUDED_DIR}/${SUBJ}_${shell}; mkdir $NEW_DIR
			cd ..
			mv $shell $NEW_DIR
			
		#If there are no dropout volumes, skip step and just copy important files to new names.
		elif (($bad_volumes <= 0)); then
			echo "${shell} shell has no signal dropout volumes" >> $notes
			cp ${shell}.bval ${shell}.${reres}mm.dropout_free.bval
			cp ${shell}.bvec ${shell}.${reres}mm.dropout_free.bvec
			cp ${shell}.${reres}mm.nii.gz ${shell}.${reres}mm.dropout_free.nii.gz

			echo "${SUBJ}		${shell}		${bad_volumes}" >> $summary_included_shells
			#echo "${SUBJ}		${shell}		${bad_volumes}" >> Step_5_summary_included_shells.txt
			echo " " >> $notes
			#echo "Performed outlier correction with FSL's eddy" >> $notes
			cd ..
		#If there are dropout volumes but the total bad volumes is less then threshold, proceed with preprocessing
		elif (($bad_volumes < $thr_vols)) && (($bad_volumes > 0)); then
			echo "Total bad volumes in ${shell} (${bad_volumes}) is less than the threshold ($thr_vols)" >> $notes
			echo "${shell} included" >> $notes

			echo "${SUBJ}		${shell}		${bad_volumes}" >> $summary_included_shells
			#echo "${SUBJ}		${shell}		${bad_volumes}" >> Step_5_summary_included_shells.txt
			echo " "
			cd ..
		fi
	fi
done

##################################################################
###    Step 4: Rerun Outlier replacement on dropout free data  ###
##################################################################


#################################
#  Set-up Shell preprocessing   #
#################################

for shell in "${shells[@]}"; do

	if [ ! -d "$shell" ]; then
		echo "No $shell directory found. Skipping preprocessing of $shell shell"
	else
	
		cd $topdir/${SUBJ}/${shell};
		#Write to preprocessing notes
		notes=${shell}.preprocessing_notes.txt
		echo " " >> $notes
		echo "Removing Dropout Volumes" >> $notes
		echo "Removing Dropout Volumes"
		if [ ! -s ${shell}.signal_dropout_vols.txt ] ; then
			echo "$shell shell has no dropout volumes. No volumes were removed." >> $notes
		else


#################################
#     Remove dropout volumes    #
#################################
		
			#Create a directory for temporary files#
			TEMP_DIR=temp_dir; mkdir $TEMP_DIR

			#split dmri image and bvec/bval files into volumes
			fslsplit ${shell}.${reres}mm.nii.gz  $TEMP_DIR/vol -t
	
			echo "Removing the following volumes from dmri, bval, and bvec files:" >> $notes

			start=0; finish=$((num_vols - 1))
			for vol in $(eval echo "{$start..$finish}"); do
		
				#Remove volumes from nii files
				if grep -q -x $vol ${shell}.signal_dropout_vols.txt; then
				echo "${vol}" >> $notes

					if (( $vol < 10 )); then 
						mv ${TEMP_DIR}/vol000${vol}.nii.gz ${TEMP_DIR}/excluded_vol000${vol}.nii.gz
					else
						mv ${TEMP_DIR}/vol00${vol}.nii.gz ${TEMP_DIR}/excluded_vol00${vol}.nii.gz
					fi 
				fi
	
			#Remove volumes from bvec and bval files
			vol2=$((vol + 1))
			if ! grep -q -x $vol ${shell}.signal_dropout_vols.txt; then
				if (( $vol2 < 10 )); then
					awk -v var=$vol2 '{print $var}' ${shell}.bval > ${TEMP_DIR}/bval0${vol2}.txt
					awk -v var=$vol2 '{print $var}' ${shell}.bvec > ${TEMP_DIR}/bvec0${vol2}.txt
				else
					awk -v var=$vol2 '{print $var}' ${shell}.bval > ${TEMP_DIR}/bval${vol2}.txt
					awk -v var=$vol2 '{print $var}' ${shell}.bvec > ${TEMP_DIR}/bvec${vol2}.txt
				fi
			fi
			done
			#Write new files
			echo "Writing included and excluded volumes nii files" >> $notes
			fslmerge -t ${shell}.${reres}mm.dropout_free ${TEMP_DIR}/vol*;
			fslmerge -t ${shell}.${reres}mm.dropout_vols ${TEMP_DIR}/excluded*

			echo "Writing new bvec and bval files" >> $notes
			paste ${TEMP_DIR}/bval* > ${shell}.${reres}mm.dropout_free.bval
			paste ${TEMP_DIR}/bvec* > ${shell}.${reres}mm.dropout_free.bvec

			rm -r $TEMP_DIR
		fi		
		cd ..
	fi
done

#######################################################
###        Set up shell preprocessing            ###
#######################################################
## changes to the correct shell subfolder
for shell in "${shells[@]}"; do

	if [ ! -d "$shell" ]; then
		echo "No $shell directory found. Skipping preprocessing of $shell shell"
	else
	
		cd $topdir/${SUBJ}/${shell};

		#Write to preprocessing notes
		notes=${shell}.preprocessing_notes.txt


#################################
#      Write Protocol Files     #
#################################
		
	
		echo " " >> $notes
		echo "Outlier Replacement/Motion Correction" >> $notes
		echo "Performing Outlier Replacement/Motion Correction on dropout_free data" >> $notes
		
		#calculate number of remaining volumes#
		removed_vols=$( sed -n '=' ${shell}.signal_dropout_vols.txt | wc -l)
		remaining_vols=$((num_vols - removed_vols))

		EDDY_DIR=eddy.outlier_detection		

		#write 2nd index file based on remaining volumes
		indx=""	
		for ((i=1; i<=$remaining_vols; i+=1)); do
		indx="$indx 1";
		done
		echo $indx > ${EDDY_DIR}/eddy_cuda_index_2.txt

		#write an slspec file that describes how the slices/MB-groups were acquired.
		start=0; finish=$((num_slices - 1))
		seq $start 2 $finish >> ${EDDY_DIR}/slspec.txt
		seq $((start + 1)) 2 $finish >> ${EDDY_DIR}/slspec.txt

		#Run eddy#
		echo "Running eddy"  >> $notes
		eddy_cuda11.2 --imain=${shell}.${reres}mm.dropout_free.nii.gz --mask=${shell}.${reres}mm_mask.nii.gz --acqp=${EDDY_DIR}/acqparams.txt --bvecs=${shell}.${reres}mm.dropout_free.bvec --index=${EDDY_DIR}/eddy_cuda_index_2.txt --bvals=${shell}.${reres}mm.dropout_free.bval --out=${EDDY_DIR}/${shell} --repol --verbose --mporder=$mporder --s2v_niter=$s2v_niter --slspec=${EDDY_DIR}/slspec.txt

#Replace the dropout free data with the new outlier free data			
		cp ${EDDY_DIR}/${shell}.nii.gz ${shell}.${reres}mm.motion_corrected.nii.gz
		cp ${EDDY_DIR}/${shell}.eddy_rotated_bvecs ${shell}.${reres}mm.motion_corrected.bvec
		cp ${shell}.${reres}mm.dropout_free.bval ${shell}.${reres}mm.motion_corrected.bval	

		echo "Performed outlier correction with FSL's eddy --repol and --mporder" >> $notes
		echo " " >> $notes
		
		cd ..
	fi
done
date


