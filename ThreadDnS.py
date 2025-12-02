
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 18:26:44 2023

@author: admin
"""

from PyQt5.QtCore import  QThread
from Generaic_functions import RGBImagePlot
import numpy as np
import traceback
import matplotlib.pyplot as plt
import datetime
import os
from scipy import ndimage
from libtiff import TIFF
import time

class DnSThread(QThread):
    def __init__(self):
        super().__init__()
        self.SampleMosaic= []
        self.sliceNum = 0
        self.tileNum = 1
        self.SnapNum = 1
        self.totalTiles = 0
        self.display_actions = 0
        
    def run(self):
        self.sliceNum = self.ui.SliceN.value()-1
        self.Imagemax = self.ui.Imagemax.value()
        self.Imagemin = self.ui.Imagemin.value()
        self.Mosaicmax = self.ui.Mosaicmax.value()
        self.Mosaicmin = self.ui.Mosaicmin.value()
        self.QueueOut()
        
    def QueueOut(self):
        self.item = self.queue.get()
        while self.item.action != 'exit':
            start=time.time()
            #self.ui.statusbar.showMessage('Display thread is doing ' + self.item.action)
            try:
                if self.item.action in ['Snap']:
                    self.display_actions += 1
                    self.Display_Snap(self.item.data)
                elif self.item.action == 'display_mosaic':
                    self.Display_mosaic(self.item.data)
                elif self.item.action == 'Clear':
                    self.SampleMosaic= []
                elif self.item.action == 'UpdateContrastImage':
                    self.Update_contrast_Image()
                elif self.item.action == 'UpdateContrastMosaic':
                    self.Update_contrast_Mosaic()
                elif self.item.action == 'display_counts':
                    self.print_display_counts()
                elif self.item.action == 'restart_tilenum':
                    self.restart_tilenum()
                elif self.item.action == 'change_slice_number':
                    self.sliceNum = self.ui.SliceN.value()-1
                    self.ui.CuSlice.setValue(self.sliceNum)
                elif self.item.action == 'WriteAgar':
                    self.WriteAgar(self.item.data, self.item.args)
                elif self.item.action == 'Init_Mosaic':
                    self.Init_Mosaic(self.item.data, self.item.args)
                elif self.item.action == 'Save_mosaic':
                    self.Save_mosaic()
                    
                else:
                    message = 'Display and save thread is doing something invalid' + self.item.action
                    self.ui.statusbar.showMessage(message)
                    # self.ui.PrintOut.append(message)
                    self.log.write(message)
                if time.time()-start>0.1:
                    print('time for DnS:',round(time.time()-start,3))
            except Exception as error:
                message = "\nAn error occurred:"+" skip the display and save action\n"
                print(message)
                self.ui.statusbar.showMessage(message)
                # self.ui.PrintOut.append(message)
                self.log.write(message)
                print(traceback.format_exc())
            # num+=1
            # print(num, 'th display\n')
            self.item = self.queue.get()
            
        self.ui.statusbar.showMessage("Display and save Thread successfully exited...")
            
    def print_display_counts(self):
        message = str(self.display_actions)+ self.ui.ACQMode.currentText() +' displayed\n'
        print(message)
        # self.ui.PrintOut.append(message)
        self.log.write(message)
        self.display_actions = 0
        
    
    def Display_Snap(self, data):
        # print(data.shape)
        scale = self.ui.scale.value()
        Xpixels = self.ui.Width.value()
        Ypixels = self.ui.Height.value()
        Zpixels = self.ui.Zstack.value()
        if data.shape[0] > 1:
            self.image=np.mean(data,0)
        else:
            self.image = data[0]
        
        pixmap = RGBImagePlot(matrix1 = np.float32(self.image[::scale, ::scale]), m=self.ui.Imagemin.value(), M=self.ui.Imagemax.value())
        # clear content on the waveformLabel
        self.ui.Image.clear()
        # update iamge on the waveformLabel
        self.ui.Image.setPixmap(pixmap)
        
        if self.ui.Save.isChecked():
            tif = TIFF.open(self.ui.DIR.toPlainText()+'/'+self.SnapFilename([Ypixels,Xpixels,Zpixels]), mode='w')
            for ii in range(Zpixels):
                # self.WriteData(data, self.AlineFilename([Yrpt,Xpixels,Zpixels]))
                tif.write_image(data[ii])
            tif.close()
            
    
    def Init_Mosaic(self, args = []):
        Xtiles = args[1][0]  # 马赛克列数
        Ytiles = args[1][1]  # 马赛克行数
        #self.scale = min(math.gcd(self.h,self.w),32) #求小于32的最大公约数作为scale
        scale = self.ui.scale.value
    
        self.SampleMosaic = np.zeros([Ytiles*(self.h//scale), Xtiles*(self.w//scale)], dtype=np.uint16)
        pixmap = RGBImagePlot(matrix1=self.SampleMosaic, m=self.ui.Mosaicmin.value(), M=self.ui.Mosaicmax.value())
        self.ui.Mosaic.clear()
        self.ui.Mosaic.setPixmap(pixmap)
        self.sliceNum += 1
    
    
    def Display_Mosaic(self, data, args = []):
        scale = self.ui.scale.value()
        Xpixels = self.ui.Width.value()
        Ypixels = self.ui.Height.value()
        Zpixels = self.ui.Zstack.value()
        if data.shape[0]> 1:
            self.image=np.mean(data,0)
        else:
            self.image = data[0]
        Xtiles = args[1][0]
        Ytiles = args[1][1]
        Y = Ytiles - args[0][1] - 1
        X = args[0][0] if args[0][1] % 2 == 1 else Xtiles - args[0][0] - 1
    
        self.SampleMosaic[self.h//scale*Y:self.h//scale*(Y+1),
                  self.w//scale*X:self.w//scale*(X+1)] = self.image[::scale, ::scale]
    
        pixmap = RGBImagePlot(matrix1=self.SampleMosaic, m=self.ui.Mosaicmin.value(), M=self.ui.Mosaicmax.value())
        self.ui.Mosaic.clear()
        self.ui.Mosaic.setPixmap(pixmap)
        if self.ui.Save.isChecked():
            tif = TIFF.open(self.ui.DIR.toPlainText()+'/mosaic/'+self.MosaicFilename([Ypixels,Xpixels,Zpixels]), mode='w')
            for ii in range(Zpixels):
                # self.WriteData(data, self.AlineFilename([Yrpt,Xpixels,Zpixels]))
                tif.write_image(data[ii])
            tif.close()
    
    # 保存马赛克拼图图像
    def Save_Mosaic(self):
        if self.ui.Save.isChecked():
            pixmap = RGBImagePlot(matrix1=self.SampleMosaic, m=self.ui.Mosaicmin.value(), M=self.ui.Mosaicmax.value())
            pixmap.save(f'{self.ui.DIR.toPlainText()}/slice{self.sliceNum}coase.tif', "TIFF")
        self.DnSBackQueue.put(self.SampleMosaic)


    def Update_contrast_Image(self):
        if self.ui.Imagemin.value() != self.Imagemin or self.ui.Imagemax.value() != self.Imagemax:
            
            try:

                pixmap = RGBImagePlot(matrix1 = np.float32(self.image), m=self.ui.Imagemin.value(), M=self.ui.Imagemax.value())
                # clear content on the waveformLabel
                self.ui.Image.clear()
                # update iamge on the waveformLabel
                self.ui.Image.setPixmap(pixmap)
            except:
                pass
            self.Imagemax = self.ui.Imagemax.value()
            self.Imagemin = self.ui.Imagemin.value()

    def Update_contrast_Mosaic(self):
        if self.ui.Mosaicmin.value() != self.Mosaicmin or self.ui.Mosaicmax.value() != self.Mosaicmax:
            try:
                pixmap = RGBImagePlot(matrix1=self.SampleMosaic, m=self.ui.Mosaicmin.value(), M=self.ui.Mosaicmax.value())
                # clear content on the waveformLabel
                self.ui.SampleMosaic.clear()
                # update iamge on the waveformLabel
                self.ui.SampleMosaic.setPixmap(pixmap)
            except:
                pass
            self.Mosaicmax = self.ui.Mosaicmax.value()
            self.Mosaicmin = self.ui.Mosaicmin.value()
            
    
    def restart_tilenum(self):
        self.tileNum = 1
        self.sliceNum = self.sliceNum+1
        self.ui.CuSlice.setValue(self.sliceNum)
        if not os.path.exists(self.ui.DIR.toPlainText()+'/mosaic/slice'+str(self.sliceNum)):
            os.mkdir(self.ui.DIR.toPlainText()+'/mosaic/slice'+str(self.sliceNum))
        # if not os.path.exists(self.ui.DIR.toPlainText()+'/surf/vol'+str(self.sliceNum)):
        #     os.mkdir(self.ui.DIR.toPlainText()+'/surf/vol'+str(self.sliceNum))
        # if not os.path.exists(self.ui.DIR.toPlainText()+'/fitting/vol'+str(self.sliceNum)):
        #     os.mkdir(self.ui.DIR.toPlainText()+'/fitting/vol'+str(self.sliceNum))
        
    def MosaicFilename(self, shape = [0,0,0]):
        filename = 'slice-'+str(self.sliceNum)+'-tile-'+str(self.tileNum)+'-Y'+str(shape[0])+'-X'+str(shape[1])+'-Z'+str(shape[2])+'.tif'
        self.tileNum = self.tileNum + 1
    
        print(filename)
        # self.ui.PrintOut.append(filename)
        self.log.write(filename)
        return filename
    
    
    def SnapFilename(self, shape):
        filename = 'Snap-'+str(self.SnapNum)+'-Y'+str(shape[0])+'-X'+str(shape[1])+'-Z'+str(shape[2])+'.tif'
        self.SnapNum = self.SnapNum + 1
        return filename
    
        
    def WriteAgar(self, data, args):
        [Ystep, Xstep] = args
        filename = 'slice-'+str(self.sliceNum)+'-agarTiles X-'+str(Xstep)+'-by Y-'+str(Ystep)+'-.bin'
        filePath = self.ui.DIR.toPlainText()
        filePath = filePath + "/" + filename
        # print(filePath)
        fp = open(filePath, 'wb')
        data.tofile(fp)
        fp.close()
        
    
