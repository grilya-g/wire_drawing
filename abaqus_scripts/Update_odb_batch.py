#!/usr/bin/env python
"""
Script for upgrading all Abaqus ODB files from input_path to input_path_updated
Run with: abaqus python Update_odb.py [batch_file input_path]
"""
import os
import sys
from abaqus import session
from abaqusConstants import CONTOURS_ON_DEF, PNG

import logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('odb_updater')

if len(sys.argv) > 2:
    # Batch mode: process only files listed in batch_file
    batch_file = sys.argv[1]
    input_path = sys.argv[2]
    with open(batch_file) as f:
        odb_list = [line.strip() for line in f if line.strip()]
    output_path = os.path.join(input_path.rstrip('/').rstrip('\\') + "_updated")
    velocities = []
else:
    # Default mode: process all files for all velocities
    velocities = [10, 20, 40]

for vel in velocities if len(sys.argv) <= 2 else [None]:
    if len(sys.argv) <= 2:
        input_path = "D:/AISI_1020/Vel_{}/".format(vel)
        output_path = input_path.rstrip('/').rstrip('\\') + "_updated"
        if not os.path.exists(input_path):
            logger.warning("Input directory does not exist: {}".format(input_path))
            continue
        odb_list = [f for f in os.listdir(input_path) if f.lower().endswith('.odb')]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        logger.info("Created output directory: {}".format(output_path))
    if not odb_list:
        logger.warning("No .odb files found in {}".format(input_path))
        continue
    for file in odb_list:
        input_file = os.path.join(input_path, file)
        output_file = os.path.join(output_path, file)
        logger.info("Upgrading ODB file: {}".format(input_file))
        try:
            session.upgradeOdb(input_file, output_file)
            logger.info("Successfully upgraded to: {}".format(output_file))
            # Screenshot part
            try:
                name = os.path.splitext(file)[0]
                o1 = session.openOdb(name=output_file)
                session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=257.88, height=198.49)
                session.viewports['Viewport: 1'].makeCurrent()
                session.viewports['Viewport: 1'].maximize()
                session.viewports['Viewport: 1'].setValues(displayedObject=o1)
                session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(CONTOURS_ON_DEF,))
                session.viewports['Viewport: 1'].view.setValues(
                    nearPlane=0.389598, farPlane=0.696669, width=0.373498, height=0.268425,
                    viewOffsetX=0.00767297, viewOffsetY=0.0200978
                )
                pic_dir = os.path.join(output_path, "Pic")
                if not os.path.exists(pic_dir):
                    os.makedirs(pic_dir)
                session.printToFile(
                    fileName=os.path.join(pic_dir, name),
                    format=PNG,
                    canvasObjects=(session.viewports['Viewport: 1'],)
                )
                o1.close()
                logger.info("Screenshot saved for: {}".format(name))
            except Exception as e:
                logger.error("Error making screenshot for {}: {}".format(file, e))
        except Exception as e:
            logger.error("Error upgrading {}: {}".format(input_file, e))

logger.info("ODB upgrade process completed")