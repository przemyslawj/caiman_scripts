sudo mkfs.ext4 -F /dev/sdb
sudo mkdir -p /mnt/disks/DATA
sudo mount /dev/sdb /mnt/disks/DATA
sudo chmod a+w /mnt/disks/DATA/
