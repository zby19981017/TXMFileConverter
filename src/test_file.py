import os
import olefile as of
import numpy as np
import array
from matplotlib import pyplot as plt
import struct
import re
from skimage.transform import resize
from gooey import Gooey, GooeyParser
from scipy.interpolate import RegularGridInterpolator
import traceback
import psutil
import sys
from vedo import Volume, show, Video
from vedo.applications import Slicer3DPlotter

dir2 = None

sys.tracebacklimit = -1


class OleReader:

    def __init__(self, inputFilename_i, outputFilename_i, previewMode_i, disableTrimming_i, trimmingSize_i,
                 targetSize_i):

        self.inputFileName = inputFilename_i
        self.outputFileName = outputFilename_i
        self.previewMode = previewMode_i
        self.disableTrimming = disableTrimming_i
        self.trimmingSize = trimmingSize_i
        self.targetSize = targetSize_i
        self.ole = None
        self.rawWidth = None
        self.rawHeight = None
        self.oriSize = None
        self.img_dir = []
        self.dataIn3D = None

    def ReadTxm(self):
        os.system('cls')

        try:
            self.ole = of.OleFileIO(self.inputFileName)
        except:
            print(r'Can not open txm file')
            exit(1)

        if self.previewMode == False:
            try:
                if self.outputFileName[-4:] != ('.csv'):
                    self.outputFileName = self.outputFileName + '.csv'
                test_f = open(self.outputFileName, 'w')
                test_f.close()
                print('Input File：' + str(self.inputFileName))
                print('Output File：' + str(self.outputFileName))
            except:
                print(r'Can not create target file')
                exit(1)

        dir1 = self.ole.listdir(self.ole)
        self.rawWidth = self.ole.openstream(['ImageInfo', 'ImageWidth'])
        self.rawWidth = self.rawWidth.read()
        self.rawWidth = struct.unpack('i', self.rawWidth)[0]
        self.rawHeight = self.ole.openstream(['ImageInfo', 'ImageHeight'])
        self.rawHeight = self.rawHeight.read()
        self.rawHeight = struct.unpack('i', self.rawHeight)[0]
        dir2 = np.array(dir1, dtype=object)
        for i in range(dir2.shape[0]):
            if re.match('ImageData', dir2[i][0]):
                self.img_dir.append(dir2[i])

        if self.rawHeight % 2 != 0 and self.rawHeight > 0:
            self.rawHeight = self.rawHeight - 1

        if self.rawWidth % 2 != 0 and self.rawWidth > 0:
            self.rawWidth = self.rawWidth - 1

        if not self.disableTrimming:
            rawSizeHeight = (self.trimmingSize[1] - self.trimmingSize[0])
            rawSizeWidth = (self.trimmingSize[3] - self.trimmingSize[2]) * 2
        else:
            rawSizeHeight = self.rawHeight
            rawSizeWidth = self.rawWidth

        self.oriSize = [len(self.img_dir), rawSizeHeight, rawSizeWidth, 0, 0]
        x1 = len(self.img_dir) * rawSizeHeight * rawSizeWidth

        x2 = int(psutil.virtual_memory()[4] / 4)

        x3 = x1 / x2
        if x3 > 0.9:
            self.oriSize[3] = int(x3) + 1
        else:
            self.oriSize[3] = 1

        if self.previewMode == True:
            try:
                print('\nStarting Preview')
                self.PreviewMode()
                print('All Done!')
                return 0
            except:
                print('\nPreview Failed\nTry to re-set the Trimming settings')
                print('Error code:', traceback.format_exc().split(',')[1][6:])
                return -1

        if not self.previewMode:
            try:
                self.dataIn3D = np.zeros(
                    np.arange(0, len(self.img_dir), self.oriSize[3], dtype=int).shape[0] * rawSizeHeight * rawSizeWidth,
                    dtype=np.float32)
            except:
                print('\nSystem memory access error')
                return -1

            try:
                print('\nStarting Convert Raw Data')
                self.ReadTXMFile()
            except:
                print('\nRead Data Failed, please contact developer')
                print('Error code:', traceback.format_exc().split(',')[1][6:])
                return -1

            try:
                print('\nStart Resizing Model')
                self.ModelResize()
                print('\nResizing Model End')
            except:
                print('\nResizing Model Failed, please contact developer')
                print('Error code:', traceback.format_exc().split(',')[1][6:])
                return -1

            try:
                if self.previewMode == False and self.dataIn3D.shape[1] > 1:
                    try:
                        print('\nStart Writing CSV File')
                        self.ExportCSVFile()
                        print('\nAll done!')
                    except:
                        print('Write CSV Data Failed, please contact developer')
                        print('Error code:', traceback.format_exc().split(',')[1][6:])
                        return 1
            except:
                raise ('Output data error')

    def PreviewMode(self):
        print('=========##################=========')
        print('Raw file size:')
        print('Depth:', len(self.img_dir),
              '\tWidth:', struct.unpack('i', self.ole.openstream(['ImageInfo', 'ImageWidth']).read())[0],
              '\tHeight:', struct.unpack('i', self.ole.openstream(['ImageInfo', 'ImageHeight']).read())[0]
              )
        print('=========##################=========')

        ole = self.ole.openstream(self.img_dir[int(len(self.img_dir) / 2)])
        img = ole.read()
        img_arr = array.array('i', img)
        img_list = img_arr.tolist()

        if (int(self.rawHeight / 2) * int(self.rawWidth / 2)) < len(img_list):
            img_list = img_list[:(int(self.rawHeight / 2) * int(self.rawWidth / 2))]

        img_list = np.reshape(img_list, (int(self.rawHeight / 2), int(self.rawWidth / 2)))

        list_number = int(self.rawWidth / 4)
        img_right = img_list[:, :list_number]
        img_left = img_list[:, list_number:]

        if (img_right.shape[1] == img_left.shape[1] - 1):
            img_right = np.column_stack((img_right, img_right[:, 0].reshape(img_right.shape[0], 1)))

        print(img_right.shape, img_left.shape)
        img_list = img_right + img_left

        if not self.disableTrimming:
            img_list = img_list[self.trimmingSize[0]:self.trimmingSize[1], self.trimmingSize[2]:self.trimmingSize[3]]
        img_list = resize(img_list, (img_list.shape[0], img_list.shape[1] * 2))
        plt.imshow(img_list)
        plt.show()

    def ReadTXMFile(self):
        if not self.disableTrimming:
            rawSizeHeight = (self.trimmingSize[1] - self.trimmingSize[0])
            rawSizeWidth = (self.trimmingSize[3] - self.trimmingSize[2]) * 2
        else:
            rawSizeHeight = self.rawHeight
            rawSizeWidth = self.rawWidth
        counterFlag = -1
        conter_img = 0

        raw_seq = np.arange(0, len(self.img_dir), self.oriSize[3], dtype=int)
        self.oriSize[4] = raw_seq.shape[0]

        for i in raw_seq:
            ole = self.ole.openstream(self.img_dir[i])
            img = ole.read()
            img_arr = array.array('i', img)
            img_list = img_arr.tolist()
            if (int(self.rawHeight / 2) * int(self.rawWidth / 2)) < len(img_list):
                img_list = img_list[:(int(self.rawHeight / 2) * int(self.rawWidth / 2))]
            img_list = np.reshape(img_list, (int(self.rawHeight / 2), int(self.rawWidth / 2)))
            img_right = img_list[:, :int(self.rawWidth / 4)]
            img_left = img_list[:, int(self.rawWidth / 4):]

            if (img_right.shape[1] == img_left.shape[1] - 1):
                img_right = np.column_stack(
                    (img_right, img_right[:, img_right.shape[1] - 1].reshape(img_right.shape[0], 1)))

            img_list = img_right + img_left
            del img_right, img_left, img_arr, img, ole
            if not self.disableTrimming:
                img_list = img_list[self.trimmingSize[0]:self.trimmingSize[1],
                           self.trimmingSize[2]:self.trimmingSize[3]]
                img_list = resize(img_list, (img_list.shape[0], img_list.shape[1] * 2))
            else:
                img_list = resize(img_list, (self.rawHeight, self.rawWidth))

            img_list = np.array(img_list, dtype=np.float32).flatten()

            baseAddress = conter_img * (rawSizeHeight * rawSizeWidth)
            self.dataIn3D[baseAddress:(baseAddress + (rawSizeWidth * rawSizeHeight))] = img_list
            del img_list

            conter_img = conter_img + 1
            counter = int(conter_img / raw_seq.shape[0] * 100)
            if counter % 10 == 0 and counterFlag != counter:
                counterFlag = counter
                print(str(conter_img) + ' / ' + str(raw_seq.shape[0]), end=' / ')
                print(counter, '%')
        print('Read raw data complete')
        return 0

    def ExportCSVFile(self):
        np.savetxt(self.outputFileName, self.dataIn3D, delimiter=",", fmt='%s')

    def ModelResize(self):
        self.oriSize[0] = self.oriSize[4]
        originalSize = self.oriSize[0:3]

        oriData_pos = np.zeros((self.targetSize[0] * self.targetSize[1] * self.targetSize[2], 3), dtype=np.float32)
        for x in np.arange(0, self.targetSize[0], 1):  # 1/10， （1/10）/10
            counter_point = 0
            baseAddress = x * (self.targetSize[1] * self.targetSize[2])
            for y in np.arange(0, self.targetSize[1], 1):
                for z in np.arange(0, self.targetSize[2], 1):
                    oriData_pos[baseAddress + counter_point] = np.float32([x, y, z])
                    counter_point = counter_point + 1

        # 还原数组
        self.dataIn3D = self.dataIn3D.reshape(originalSize)

        temp_lx = np.linspace(0, self.targetSize[0], originalSize[0])
        temp_ly = np.linspace(0, self.targetSize[1], originalSize[1])
        temp_lz = np.linspace(0, self.targetSize[2], originalSize[2])

        interpolatingFunction = RegularGridInterpolator((temp_lx, temp_ly, temp_lz), self.dataIn3D)
        resizedPointMatrix = interpolatingFunction((oriData_pos[:, 0], oriData_pos[:, 1], oriData_pos[:, 2]))

        pointsNumTotal = resizedPointMatrix.shape[0]

        maxPoint = resizedPointMatrix.max()
        minPoint = resizedPointMatrix.min()
        counterFlag = -1
        for xx in range(pointsNumTotal):
            temp = resizedPointMatrix[xx]
            resizedPointMatrix[xx] = np.float32((temp - minPoint) * pow((maxPoint - minPoint), -1))
            # print(xx)
            counter = int(xx / pointsNumTotal * 100)
            if counter % 10 == 0 and counterFlag != counter:
                counterFlag = counter
                print(xx, ' / ', pointsNumTotal, ' / ', counter, '%')

        resizedPointMatrix = resizedPointMatrix.reshape(self.targetSize[0], self.targetSize[1], self.targetSize[2])

        self.dataIn3D = None

        self.dataIn3D = np.zeros((self.targetSize[0] * self.targetSize[1] * self.targetSize[2], 4), dtype=np.float32)

        for z in range(self.targetSize[2]):
            counter_point = 0
            baseAddress = z * (self.targetSize[1] * self.targetSize[0])
            for y in range(self.targetSize[1]):
                for x in range(self.targetSize[0]):
                    self.dataIn3D[baseAddress + counter_point] = ([x, y, z, resizedPointMatrix[x, y, z]])
                    resizedPointMatrix[x, y, z] = 0
                    counter_point = counter_point + 1


def main():
    # inputFilename = 'E:\\备份\\数据\\蒲老师岩心数据\\thr_water-thr_recon.txm'
    # inputFilename = 'E:\\备份\\数据\\蒲老师岩心数据\\sec_water-sec_recon.txm'
    inputFilename = 'E:\\备份\\数据\\CT-2023-4\\JY-4\\JY-4_dry_dry_recon.txm'
    outputFilename = './TEST.csv'

    previewMode = False
    disableTrimming = True
    T_size = [288, 478, 140, 239]
    I_size = [100, 100, 250]

    txmReader = OleReader(inputFilename, outputFilename, previewMode, disableTrimming, T_size,
                          I_size)

    status = txmReader.ReadTxm()

    try:
        if status == 0 or status == -1:
            raise ('Program error return code', status)
    except:
        None


if __name__ == '__main__':
    main()
