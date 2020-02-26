#!/bin/bash
# ./pipeline.sh animals_array dates_array
# example: ./pipeline --animals "E-BL E-TR" --dates "2019-08-27" --dry_run

EXP_MONTH=2019-08
EXP_TITLE=habituation
dry_run=0
cnmfe_only=0


while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --dry_run)
        dry_run=1
        shift # past argument
        ;;
        --cnmfe_only)
        cnmfe_only=1
        shift # past argument
        ;;
        --animals)
        animals="$2"
        shift # past argument
        shift # past value
        ;;
        --dates)
        dates="$2"
        shift # past argument
        shift # past value
        ;;
        --exp_title)
        EXP_TITLE="$2"
        shift # past argument
        shift # past value
        ;;
        --exp_month)
        EXP_MONTH="$2"
        shift # past argument
        shift # past value
        ;;
    esac
done

echo "Starting pipeline for exp_month=${EXP_MONTH} exp_title=${EXP_TITLE}"
echo "                      animals=${animals} dates=${dates}"

#conda activate caiman
for exp_date in ${dates[*]}; do
    for animal in ${animals[*]}; do
        export ANIMAL=$animal
        export EXP_DATE=$exp_date
        export EXP_MONTH
        export EXP_TITLE
        # load defaults
        source vars_setup.sh
        echo "Running pipeline for animal=${ANIMAL} date=${EXP_DATE}"

        if [ $dry_run -eq 1 ]; then
            continue
        fi

        if [ $cnmfe_only -eq 0 ]; then
            ./gdrive_download.sh

            python downsample.py
            status=$?
            if [ $status -ne 0 ]; then
                exit $status
            fi

            time python caiman_mc.py
            status=$?
            if [ $status -ne 0 ]; then
                exit $status
            fi

            python memmap_mc_files.py
            status=$?
            if [ $status -ne 0 ]; then
                exit $status
            fi

            python create_sessions_info.py
            status=$?
            if [ $status -ne 0 ]; then
                exit $status
            fi
        fi

        if [ $cnmfe_only -eq 1 ]; then
            ./gdrive_download_processed.sh
        fi

        time python run_cnmfe.py
        status=$?
        if [ $status -ne 0 ]; then
            echo "Run CNMFE exited with error"
            exit $status
        fi

        ./gdrive_upload.sh

        rm -rf ${LOCAL_ROOTDIR}/${DOWNSAMPLE_SUBPATH}/${EXP_MONTH}/${EXP_TITLE}/${EXP_DATE}
    done

done

