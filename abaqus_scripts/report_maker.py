import glob
from abaqus import *
from abaqusConstants import *
import __main__
import section
import regionToolset
import displayGroupMdbToolset as dgm
import part
import material
import assembly
import step
import interaction
import load
import mesh
import optimization
import job
import sketch
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
import connectorBehavior
import shutil
import os
from os import listdir
from os.path import isfile, join
import math
import numpy as np

# To run this script from cmd use the following line:
# abaqus cae noGUI=report_maker.py

# script_dir_path = os.path.dirname(os.path.realpath(__file__))
# path_import = script_dir_path
path_import = 'D:\\odb_20_30\\'
path_import = 'E://Abaqus_Ilya//work_dir//jobs//'
path_import = 'd:\\odb_20_30\\back_danya\\'
path_import = r'C:\temp//'
path_import = r'D:\odb_20_30\ann_test\\'

# path_export = os.path.dirname(os.path.realpath(__file__))
path_export = "D:\\odb_20_30\\reports_20_30\\"
path_export = path_import + "\\reports_20_30\\"

if not os.path.exists(path_export):
    os.makedirs(path_export)

def for_report(list_of_odb):
    for odb_file in list_of_odb:    
        temp_odb_view = session.openOdb(name = path_import + odb_file)
        session.viewports['Viewport: 1'].setValues(displayedObject=temp_odb_view)
        odb = session.odbs[path_import + odb_file]
        session.viewports['Viewport: 1'].odbDisplay.setFrame(step='Step-1', frame=odb.steps['Step-1'].frames.__len__()-1)

        session.fieldReportOptions.setValues(printTotal=OFF,printMinMax=OFF)

        session.writeFieldReport(fileName= path_export + odb_file[:-4]+'.txt', append=OFF, sortItem='Nodal Label', odb=odb,
        step=0, frame=odb.steps['Step-1'].frames.__len__()-1, outputPosition=NODAL,
        variable=(
        ('COORD',NODAL,((COMPONENT, 'COOR1'),(COMPONENT, 'COOR2'),)),
        ('S', INTEGRATION_POINT, ((COMPONENT, 'S11'), (COMPONENT, 'S22'), (COMPONENT, 'S33'),(COMPONENT, 'S12'), )),
        ('PE', INTEGRATION_POINT, ((COMPONENT, 'PE11'), (COMPONENT, 'PE22'), (COMPONENT, 'PE33'),(COMPONENT, 'PE12'), )),
        ('LE', INTEGRATION_POINT, ((COMPONENT, 'LE11'), (COMPONENT, 'LE22'), (COMPONENT, 'LE33'),(COMPONENT, 'LE12'),),),
        ))

        odb.close()

rep_list = [f for f in listdir(path_import) if isfile(join(path_import, f)) and f.endswith('.odb')]
 
print(rep_list)
for_report(rep_list)

