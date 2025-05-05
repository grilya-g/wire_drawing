from abaqus import *
from abaqusConstants import *

session.Viewport(
    name="Viewport: 1",
    origin=(0.0, 0.0),
    width=257.887481689453,
    height=198.488891601562,
)
session.viewports["Viewport: 1"].makeCurrent()
session.viewports["Viewport: 1"].maximize()
from caeModules import *
from driverUtils import executeOnCaeStartup

executeOnCaeStartup()

for vel in [10, 20, 40]:
    input_path = "D:/AISI_1020/Vel_{}_updated/".format(vel)
    pics_path = "D:/AISI_1020/Vel_{}_updated/Pic/".format(vel)

    name_odb = []

    for file in os.listdir(input_path):
        if file.endswith(".odb"):
            name_odb.append(file.split(".")[0])


    for name in name_odb:
        if os.path.exists(input_path + name + ".odb"):
            session.viewports["Viewport: 1"].assemblyDisplay.setValues(
                optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF
            )
            o1 = session.openOdb(name=input_path + name + ".odb")
            session.viewports["Viewport: 1"].setValues(displayedObject=o1)
            session.viewports["Viewport: 1"].odbDisplay.display.setValues(
                plotState=(CONTOURS_ON_DEF,)
            )
            session.viewports["Viewport: 1"].view.setValues(
                nearPlane=0.389598,
                farPlane=0.696669,
                width=0.373498,
                height=0.268425,
                viewOffsetX=0.00767297,
                viewOffsetY=0.0200978,
            )
            session.printToFile(
                fileName=pics_path + name,
                format=PNG,
                canvasObjects=(session.viewports["Viewport: 1"],),
            )
        else:
            pass
