import os
import csv

def process_files(parent_directory):
    # Create a list to hold all the data
    data = [['SubjectID', 'B1000 Bad Volumes', 'B2000 Bad Volumes', 'B1000 Bad Slices per Volume', 'B2000 Bad Slices per Volume']]

    # Walk through all directories and subdirectories in the parent directory
    for root, dirs, files in os.walk(parent_directory):
        if 'Signal_Dropout.excluded_vols.txt' in files:
            # Extract SubjectID from the filepath
            subject_id = os.path.normpath(root).split(os.sep)[-2]

            with open(os.path.join(root, 'Signal_Dropout.excluded_vols.txt'), 'r') as file:
                lines = file.readlines()

                b1000_bad_volumes = ''
                b2000_bad_volumes = ''
                b1000_bad_slices_per_volume = ''
                b2000_bad_slices_per_volume = ''

                for line in lines[1:]:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        shell = parts[1]
                        bad_volumes_count = parts[2]
                        bad_slices_count_list = ' '.join(parts[4:])

                        if shell == 'b1000':
                            b1000_bad_volumes += bad_volumes_count + ','
                            b1000_bad_slices_per_volume += bad_slices_count_list + ','
                        elif shell == 'b2000':
                            b2000_bad_volumes += bad_volumes_count + ','
                            b2000_bad_slices_per_volume += bad_slices_count_list + ','

                data.append([subject_id, b1000_bad_volumes.strip(','), b2000_bad_volumes.strip(','), b1000_bad_slices_per_volume.strip(','), b2000_bad_slices_per_volume.strip(',')])

    # Write data to CSV file in parent directory
    with open(os.path.join(parent_directory, 'results.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

