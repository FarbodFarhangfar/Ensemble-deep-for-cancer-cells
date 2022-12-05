import tarfile

my_tar = tarfile.open("E:/ensmeble/msdnet-step=7-block=5.pth.tar", 'w')
my_tar.extractall("E:/ensmeble/msdnet_pretrained/msdnet-step=7-block=5.pth", 'r')
my_tar.close()