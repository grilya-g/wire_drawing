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
import math
import time
from odbAccess import*
from abaqusConstants import*
from odbMaterial import*
from odbSection import*

def r(job_name):
    # found = re.search('red_(.+?)_', job_name).group(1)
    splitted_job = job_name.split('_')
    for i in range(len(splitted_job)):
        if splitted_job[i] == 'red':
            found = splitted_job[i+1]
    reduct = float(found) / 10000
    return reduct

def c(job_name):
    splitted_job = job_name.split('_')
    for i in range(len(splitted_job)):
        if splitted_job[i] == 'cal':
            found = splitted_job[i+1]
    cali = float(found) / 100
    return cali

def fric(job_name):
    splitted_job = job_name.split('_')
    for i in range(len(splitted_job)):
        if splitted_job[i] == 'fric':
            found = splitted_job[i+1][1:]
    fr = float(found) / 1000
    return fr

def v(job_name):
    splitted_job = job_name.split('_')
    for i in range(len(splitted_job)):
        if splitted_job[i] == 'vel':
            found = splitted_job[i+1]
    velo = int(found)
    return velo

def h(job_name):
    splitted_job = job_name.split('_')
    for i in range(len(splitted_job)):
        if splitted_job[i] == '2a':
            found = splitted_job[i+1]
    ha = int(found) / 2
    return ha

NUM_CPUS = 2
def do_calc_double_plus_pack(job_name, NUM_CPUS=NUM_CPUS):
    friction = fric(job_name)

    idx = job_to_calc_name.find('f') - 1
    label = job_to_calc_name[:idx]
    
        # interactionProperties    tangentialBehavior
    mdb.models[label].interactionProperties['IntProp-1'].tangentialBehavior.setValues(
        dependencies=0, directionality=ISOTROPIC, elasticSlipStiffness=None, 
        formulation=PENALTY, fraction=0.005, maximumElasticSlip=FRACTION, 
        pressureDependency=OFF, shearStressLimit=None, slipRateDependency=OFF, 
        table=((friction, ), ), temperatureDependency=OFF)
        
    # interactionProperties NormalBehavior
    mdb.models[label].interactionProperties['IntProp-1'].NormalBehavior(
        allowSeparation=ON, constraintEnforcementMethod=DEFAULT, 
        pressureOverclosure=HARD) 
    
    job = mdb.Job(name= job_name, model= label ,nodalOutputPrecision=FULL, numCpus=NUM_CPUS,numDomains=NUM_CPUS,userSubroutine=None)
    mdb.jobs[job_name].setValues(explicitPrecision=DOUBLE_PLUS_PACK)
    mdb.jobs[job_name].writeInput(consistencyChecking=OFF)
    job.submit()
    job.waitForCompletion()

def check_if_no_error(job_name, job_list):
    with open(job_name+'.dat') as f:
        if 'THE ANALYSIS HAS BEEN COMPLETED' in f.read():
            return True
        else:
            return False

odbPath = r'D:\odb_20_30\ann_test'

os.chdir(odbPath)

# Detects all jobs with fric=100 and 25
# redo_list = [f[:-4] for f in os.listdir(odbPath) 
            # if os.path.isfile(os.path.join(odbPath, f)) and (f.endswith('100.odb') or f.endswith('25.odb'))]



all_job_to_calc_list = [
"zeides_2a_28_red_1700_cal_40_vel_20_fric_075"  ,
# "zeides_2a_36_red_1700_cal_40_vel_20_fric_075"  ,
# "zeides_2a_44_red_1700_cal_40_vel_20_fric_075"  ,
# "zeides_2a_28_red_1700_cal_90_vel_20_fric_075"  ,
# "zeides_2a_36_red_1700_cal_90_vel_20_fric_075"  ,
# "zeides_2a_44_red_1700_cal_90_vel_20_fric_075"  ,
# "zeides_2a_28_red_1700_cal_125_vel_20_fric_075" ,
# "zeides_2a_36_red_1700_cal_125_vel_20_fric_075" ,
# "zeides_2a_44_red_1700_cal_125_vel_20_fric_075" ,
"zeides_2a_28_red_1900_cal_40_vel_20_fric_075"  ,
"zeides_2a_36_red_1900_cal_40_vel_20_fric_075"  ,
# "zeides_2a_44_red_1900_cal_40_vel_20_fric_075"  ,
"zeides_2a_28_red_1900_cal_90_vel_20_fric_075"  ,
# "zeides_2a_36_red_1900_cal_90_vel_20_fric_075"  ,
# "zeides_2a_44_red_1900_cal_90_vel_20_fric_075"  ,
# "zeides_2a_28_red_1900_cal_125_vel_20_fric_080" ,
# "zeides_2a_36_red_1900_cal_125_vel_20_fric_080" ,
"zeides_2a_44_red_1900_cal_125_vel_20_fric_080" 
]

jd = [
'zeides_2a_8_red_2000_cal_10_vel_30_fric_025',
'zeides_2a_8_red_2000_cal_10_vel_30_fric_0100',
'zeides_2a_8_red_2000_cal_10_vel_30_fric_025'
]

for x in all_job_to_calc_list:
  if x in jd:
    all_job_to_calc_list.remove(x)

# while all_job_to_calc_list:
i = 0
for job_to_calc_name in all_job_to_calc_list:
    print(len(all_job_to_calc_list) - i, 'jobs left')
    do_calc_double_plus_pack(job_to_calc_name)
    if check_if_no_error(job_to_calc_name, all_job_to_calc_list):
        all_job_to_calc_list.remove(job_to_calc_name)
    i += 1
    