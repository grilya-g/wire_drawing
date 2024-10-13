# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2022 replay file
# Internal Version: 2021_09_15-20.57.30 176069
# Run by demin on Sat Mar 25 14:23:52 2023
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=125.413887023926, 
    height=60.5777816772461)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from caeModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
mdb.models.changeKey(fromName='Model-1', toName='Zeides_config')
session.viewports['Viewport: 1'].setValues(displayedObject=None)
s = mdb.models['Zeides_config'].ConstrainedSketch(name='__profile__', 
    sheetSize=200.0)
g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
s.sketchOptions.setValues(viewStyle=AXISYM)
s.setPrimaryObject(option=STANDALONE)
s.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, 100.0))
s.FixedConstraint(entity=g[2])
session.viewports['Viewport: 1'].view.setValues(nearPlane=94.2393, 
    farPlane=94.3225, width=0.222102, height=0.111361, cameraPosition=(
    0.0437342, 0.0289055, 94.2809), cameraTarget=(0.0437342, 0.0289055, 0))
s.Line(point1=(0.018, 0.0), point2=(0.0, 0.0))
s.HorizontalConstraint(entity=g[3], addUndoState=False)
s.CoincidentConstraint(entity1=v[1], entity2=g[2], addUndoState=False)
s.Line(point1=(0.0, 0.0), point2=(0.0, 0.11))
s.VerticalConstraint(entity=g[4], addUndoState=False)
s.PerpendicularConstraint(entity1=g[3], entity2=g[4], addUndoState=False)
s.Line(point1=(0.0, 0.11), point2=(0.018, 0.11))
s.HorizontalConstraint(entity=g[5], addUndoState=False)
s.PerpendicularConstraint(entity1=g[4], entity2=g[5], addUndoState=False)
s.Line(point1=(0.018, 0.11), point2=(0.018, 0.0))
s.VerticalConstraint(entity=g[6], addUndoState=False)
s.PerpendicularConstraint(entity1=g[5], entity2=g[6], addUndoState=False)
session.viewports['Viewport: 1'].view.setValues(nearPlane=94.2164, 
    farPlane=94.3454, width=0.389202, height=0.195144, cameraPosition=(
    0.0655333, 0.0561067, 94.2809), cameraTarget=(0.0655333, 0.0561067, 0))
p = mdb.models['Zeides_config'].Part(name='Wire', dimensionality=AXISYMMETRIC, 
    type=DEFORMABLE_BODY)
p = mdb.models['Zeides_config'].parts['Wire']
p.BaseShell(sketch=s)
s.unsetPrimaryObject()
p = mdb.models['Zeides_config'].parts['Wire']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
del mdb.models['Zeides_config'].sketches['__profile__']
p = mdb.models['Zeides_config'].parts['Wire']
s1 = p.features['Shell planar-1'].sketch
mdb.models['Zeides_config'].ConstrainedSketch(name='__edit__', objectToCopy=s1)
s2 = mdb.models['Zeides_config'].sketches['__edit__']
g, v, d, c = s2.geometry, s2.vertices, s2.dimensions, s2.constraints
s2.setPrimaryObject(option=SUPERIMPOSE)
p.projectReferencesOntoSketch(sketch=s2, 
    upToFeature=p.features['Shell planar-1'], filter=COPLANAR_EDGES)
s2.Line(point1=(0.0, 0.055), point2=(0.0179999999618158, 0.055))
s2.HorizontalConstraint(entity=g[7], addUndoState=False)
s2.PerpendicularConstraint(entity1=g[4], entity2=g[7], addUndoState=False)
s2.CoincidentConstraint(entity1=v[4], entity2=g[4], addUndoState=False)
s2.EqualDistanceConstraint(entity1=v[1], entity2=v[2], midpoint=v[4], 
    addUndoState=False)
s2.CoincidentConstraint(entity1=v[5], entity2=g[6], addUndoState=False)
s2.EqualDistanceConstraint(entity1=v[3], entity2=v[0], midpoint=v[5], 
    addUndoState=False)
s2.VerticalDimension(vertex1=v[4], vertex2=v[1], textPoint=(
    -0.0217217765748501, 0.00815462693572044), value=0.055)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.20772, 
    farPlane=0.238132, width=0.0811189, height=0.0406726, cameraPosition=(
    0.0178093, 0.0532788, 0.222926), cameraTarget=(0.0178093, 0.0532788, 0))
s2.delete(objectList=(g[7], ))
s2.unsetPrimaryObject()
p = mdb.models['Zeides_config'].parts['Wire']
p.features['Shell planar-1'].setValues(sketch=s2)
del mdb.models['Zeides_config'].sketches['__edit__']
p = mdb.models['Zeides_config'].parts['Wire']
p.regenerate()
p = mdb.models['Zeides_config'].parts['Wire']
p.regenerate()
p = mdb.models['Zeides_config'].parts['Wire']
f, e, d1 = p.faces, p.edges, p.datums
t = p.MakeSketchTransform(sketchPlane=f[0], sketchPlaneSide=SIDE1, origin=(
    0.009, 0.055, 0.0))
s = mdb.models['Zeides_config'].ConstrainedSketch(name='__profile__', 
    sheetSize=0.222, gridSpacing=0.005, transform=t)
g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
s.sketchOptions.setValues(decimalPlaces=3)
s.setPrimaryObject(option=SUPERIMPOSE)
p = mdb.models['Zeides_config'].parts['Wire']
p.projectReferencesOntoSketch(sketch=s, filter=COPLANAR_EDGES)
s.Line(point1=(0.009, 0.0), point2=(-0.00900000003911555, 0.0))
s.HorizontalConstraint(entity=g[7], addUndoState=False)
s.PerpendicularConstraint(entity1=g[2], entity2=g[7], addUndoState=False)
s.CoincidentConstraint(entity1=v[4], entity2=g[2], addUndoState=False)
s.EqualDistanceConstraint(entity1=v[0], entity2=v[1], midpoint=v[4], 
    addUndoState=False)
s.CoincidentConstraint(entity1=v[5], entity2=g[4], addUndoState=False)
s.EqualDistanceConstraint(entity1=v[2], entity2=v[3], midpoint=v[5], 
    addUndoState=False)
p = mdb.models['Zeides_config'].parts['Wire']
f = p.faces
pickedFaces = f.getSequenceFromMask(mask=('[#1 ]', ), )
e1, d2 = p.edges, p.datums
p.PartitionFaceBySketch(faces=pickedFaces, sketch=s)
s.unsetPrimaryObject()
del mdb.models['Zeides_config'].sketches['__profile__']
p = mdb.models['Zeides_config'].parts['Wire']
e = p.edges
edges = e.getSequenceFromMask(mask=('[#20 ]', ), )
p.Set(edges=edges, name='Fix_wire')
#: The set 'Fix_wire' has been created (1 edge).
p = mdb.models['Zeides_config'].parts['Wire']
f = p.faces
faces = f.getSequenceFromMask(mask=('[#3 ]', ), )
p.Set(faces=faces, name='Full_wire')
#: The set 'Full_wire' has been created (2 faces).
p = mdb.models['Zeides_config'].parts['Wire']
s = p.edges
side1Edges = s.getSequenceFromMask(mask=('[#66 ]', ), )
p.Surface(side1Edges=side1Edges, name='For_contact')
#: The surface 'For_contact' has been created (4 edges).
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.241461, 
    farPlane=0.291426, width=0.134066, height=0.0635933, 
    viewOffsetX=0.00171179, viewOffsetY=-6.31467e-05)
session.viewports['Viewport: 1'].partDisplay.setValues(mesh=ON)
session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
    meshTechnique=ON)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=OFF)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.231152, 
    farPlane=0.301735, width=0.210545, height=0.10026, viewOffsetX=0.0193136, 
    viewOffsetY=-0.00838997)
p = mdb.models['Zeides_config'].parts['Wire']
e = p.edges
pickedEdges = e.getSequenceFromMask(mask=('[#20 ]', ), )
p.seedEdgeByNumber(edges=pickedEdges, number=20, constraint=FINER)
p = mdb.models['Zeides_config'].parts['Wire']
p.seedPart(size=0.0055, deviationFactor=0.1, minSizeFactor=0.1)
p = mdb.models['Zeides_config'].parts['Wire']
p.seedPart(size=0.00095, deviationFactor=0.1, minSizeFactor=0.1)
p = mdb.models['Zeides_config'].parts['Wire']
p.generateMesh()
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.23994, 
    farPlane=0.292946, width=0.141725, height=0.0674883, viewOffsetX=0.0144776, 
    viewOffsetY=-0.00573461)
session.viewports['Viewport: 1'].partDisplay.setValues(mesh=OFF)
session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
    meshTechnique=OFF)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
session.viewports['Viewport: 1'].partDisplay.setValues(mesh=ON)
session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
    meshTechnique=ON)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=OFF)
session.viewports['Viewport: 1'].partDisplay.setValues(mesh=OFF)
session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
    meshTechnique=OFF)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.260993, 
    farPlane=0.271894, width=0.0291147, height=0.0138104, 
    viewOffsetX=0.00198959, viewOffsetY=-0.00630356)
session.viewports['Viewport: 1'].partDisplay.setValues(mesh=ON)
session.viewports['Viewport: 1'].partDisplay.setValues(mesh=OFF)
p = mdb.models['Zeides_config'].parts['Wire']
e = p.edges
edges = e.getSequenceFromMask(mask=('[#1 ]', ), )
p.Set(edges=edges, name='For_results')
#: The set 'For_results' has been created (1 edge).
s1 = mdb.models['Zeides_config'].ConstrainedSketch(name='__profile__', 
    sheetSize=200.0)
g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
s1.sketchOptions.setValues(viewStyle=AXISYM)
s1.setPrimaryObject(option=STANDALONE)
s1.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, 100.0))
s1.FixedConstraint(entity=g[2])
session.viewports['Viewport: 1'].view.setValues(nearPlane=94.2789, 
    farPlane=94.2829, width=0.0107107, height=0.00537028, cameraPosition=(
    0.00123194, -0.000871561, 94.2809), cameraTarget=(0.00123194, -0.000871561, 
    0))
s1.Line(point1=(0.001, -0.001), point2=(0.001, 0.0))
s1.VerticalConstraint(entity=g[3], addUndoState=False)
session.viewports['Viewport: 1'].view.setValues(nearPlane=94.2773, 
    farPlane=94.2845, width=0.0190112, height=0.00953212, cameraPosition=(
    0.00199629, 0.00149908, 94.2809), cameraTarget=(0.00199629, 0.00149908, 0))
s1.Line(point1=(0.001, 0.0), point2=(0.00165515905246139, 0.002500040223822))
s1.Line(point1=(0.001, -0.001), point2=(0.00148920295760036, 
    -0.00157728116028011))
s1.AngularDimension(line1=g[4], line2=g[3], textPoint=(0.00154452165588737, 
    0.000351181486621499), value=172.0)
p = mdb.models['Zeides_config'].Part(name='Die_2a_8_d0_36mm', 
    dimensionality=AXISYMMETRIC, type=ANALYTIC_RIGID_SURFACE)
p = mdb.models['Zeides_config'].parts['Die_2a_8_d0_36mm']
p.AnalyticRigidSurf2DPlanar(sketch=s1)
s1.unsetPrimaryObject()
p = mdb.models['Zeides_config'].parts['Die_2a_8_d0_36mm']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
del mdb.models['Zeides_config'].sketches['__profile__']
p = mdb.models['Zeides_config'].parts['Die_2a_8_d0_36mm']
v1, e, d1, n = p.vertices, p.edges, p.datums, p.nodes
p.ReferencePoint(point=v1[0])
p = mdb.models['Zeides_config'].parts['Die_2a_8_d0_36mm']
r = p.referencePoints
refPoints=(r[2], )
p.Set(referencePoints=refPoints, name='Vel_RP')
#: The set 'Vel_RP' has been created (1 reference point).
p = mdb.models['Zeides_config'].parts['Die_2a_8_d0_36mm']
s = p.edges
side1Edges = s.getSequenceFromMask(mask=('[#7 ]', ), )
p.Surface(side1Edges=side1Edges, name='For_contact')
#: The surface 'For_contact' has been created (3 edges).
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON, 
    engineeringFeatures=ON)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=OFF)
mdb.models['Zeides_config'].Material(name='Steel_A12')
mdb.models['Zeides_config'].materials['Steel_A12'].Density(table=((7870.0, ), 
    ))
mdb.models['Zeides_config'].materials['Steel_A12'].Elastic(table=((
    200000000000.0, 0.29), ))
    
mdb.models['Zeides_config'].materials['Steel_A12'].Plastic(scaleStress=None, 
    table=(
    (1000.0, 0.0),
    (82372287.6437224, 0.0517241379310345),
    (101377041.05285312, 0.103448275862069),
    (114466642.17152564, 0.1551724137931035),
    (124766529.45567553, 0.206896551724138),
    (133389806.84695707, 0.2586206896551725),
    (140876134.61454436, 0.310344827586207),
    (147532598.03474334, 0.36206896551724155),
    (153552389.28603396, 0.413793103448276),
    (159065779.82507503, 0.4655172413793105),
    (164165210.31010458, 0.517241379310345),
    (168918896.8686454, 0.5689655172413796),
    (173378767.18875283, 0.620689655172414),
    (177585360.56424478, 0.6724137931034485),
    (181570995.23922333, 0.7241379310344831),
    (185361894.47256643, 0.7758620689655176),
    (188979659.51538423, 0.827586206896552),
    (192442317.9651268, 0.8793103448275865),
    (195765087.4184464, 0.931034482758621),
    (198960943.09465662, 0.9827586206896556),
    (202041047.31242186, 1.03448275862069),
    (205015079.59716716, 1.0862068965517246),
    (207891494.00005025, 1.1379310344827591),
    (210677722.22465163, 1.1896551724137936),
    (213380335.81160077, 1.241379310344828),
    (216005176.979257, 1.2931034482758625),
    (218557465.1777828, 1.344827586206897),
    (221041884.61687282, 1.3965517241379315),
    (223462656.73704347, 1.4482758620689662)
    )
    )
mdb.models['Zeides_config'].HomogeneousSolidSection(name='Section-1', 
    material='Steel_A12', thickness=None)
p1 = mdb.models['Zeides_config'].parts['Wire']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
p = mdb.models['Zeides_config'].parts['Wire']
region = p.sets['Full_wire']
p = mdb.models['Zeides_config'].parts['Wire']
p.SectionAssignment(region=region, sectionName='Section-1', offset=0.0, 
    offsetType=MIDDLE_SURFACE, offsetField='', 
    thicknessAssignment=FROM_SECTION)
a = mdb.models['Zeides_config'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    adaptiveMeshConstraints=ON, optimizationTasks=OFF, 
    geometricRestrictions=OFF, stopConditions=OFF)
mdb.models['Zeides_config'].StaticStep(name='Step-1', previous='Initial', 
    timePeriod=9.0, maxNumInc=999999999, initialInc=1e-11, minInc=1e-13, 
    maxInc=0.001, nlgeom=ON)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Step-1')
mdb.models['Zeides_config'].fieldOutputRequests['F-Output-1'].setValues(
    variables=('S', 'MISES', 'E', 'PE', 'PEEQ', 'PEMAG', 'EE', 'NE', 'LE', 'U', 
    'RF', 'CF', 'CSTRESS', 'CDISP', 'EVOL', 'COORD'), frequency=10)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(interactions=ON, 
    constraints=ON, connectors=ON, engineeringFeatures=ON, 
    adaptiveMeshConstraints=OFF)
mdb.models['Zeides_config'].ContactProperty('IntProp-1')
mdb.models['Zeides_config'].interactionProperties['IntProp-1'].TangentialBehavior(
    formulation=PENALTY, directionality=ISOTROPIC, slipRateDependency=OFF, 
    pressureDependency=OFF, temperatureDependency=OFF, dependencies=0, table=((
    0.05, ), ), shearStressLimit=None, maximumElasticSlip=FRACTION, 
    fraction=0.005, elasticSlipStiffness=None)
mdb.models['Zeides_config'].interactionProperties['IntProp-1'].NormalBehavior(
    pressureOverclosure=HARD, allowSeparation=ON, 
    constraintEnforcementMethod=DEFAULT)
#: The interaction property "IntProp-1" has been created.
session.viewports['Viewport: 1'].assemblyDisplay.setValues(interactions=OFF, 
    constraints=OFF, connectors=OFF, engineeringFeatures=OFF)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.257117, 
    farPlane=0.275769, width=0.0498627, height=0.0236521, 
    viewOffsetX=0.00728794, viewOffsetY=-0.0485076)
a = mdb.models['Zeides_config'].rootAssembly
a.DatumCsysByThreePoints(coordSysType=CYLINDRICAL, origin=(0.0, 0.0, 0.0), 
    point1=(1.0, 0.0, 0.0), point2=(0.0, 0.0, -1.0))
p = mdb.models['Zeides_config'].parts['Die_2a_8_d0_36mm']
a.Instance(name='Die_2a_8_d0_36mm-1', part=p, dependent=ON)
p = mdb.models['Zeides_config'].parts['Wire']
a.Instance(name='Wire-1', part=p, dependent=ON)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.264687, 
    farPlane=0.275702, width=0.0294198, height=0.0139551, 
    viewOffsetX=-0.00437851, viewOffsetY=-0.0527398)
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=OFF, 
    engineeringFeatures=OFF)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
p1 = mdb.models['Zeides_config'].parts['Die_2a_8_d0_36mm']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
p = mdb.models['Zeides_config'].parts['Die_2a_8_d0_36mm']
s = p.features['2D Analytic rigid shell-1'].sketch
mdb.models['Zeides_config'].ConstrainedSketch(name='__edit__', objectToCopy=s)
s2 = mdb.models['Zeides_config'].sketches['__edit__']
g, v, d, c = s2.geometry, s2.vertices, s2.dimensions, s2.constraints
s2.setPrimaryObject(option=SUPERIMPOSE)
p.projectReferencesOntoSketch(sketch=s2, 
    upToFeature=p.features['2D Analytic rigid shell-1'], filter=COPLANAR_EDGES)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.223514, 
    farPlane=0.228974, width=0.0145642, height=0.00730242, cameraPosition=(
    0.00134833, 0.000268612, 0.226244), cameraTarget=(0.00134833, 0.000268612, 
    0))
s2.DistanceDimension(entity1=g[3], entity2=g[2], textPoint=(
    0.000776213360950351, 0.00166859524324536), value=0.01)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.223667, 
    farPlane=0.228821, width=0.0137464, height=0.00689236, cameraPosition=(
    0.00726866, 0.00300554, 0.226244), cameraTarget=(0.00726866, 0.00300554, 
    0))
s2.ObliqueDimension(vertex1=v[1], vertex2=v[2], textPoint=(0.00795531086623669, 
    0.00212241103872657), value=0.004)
d[2].setValues(value=0.006, )
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.22293, 
    farPlane=0.229558, width=0.0200076, height=0.0100317, cameraPosition=(
    0.00626168, 0.0019472, 0.226244), cameraTarget=(0.00626168, 0.0019472, 0))
s2.ObliqueDimension(vertex1=v[0], vertex2=v[1], textPoint=(0.00893000233918428, 
    -0.000710524618625641), value=0.005)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.223116, 
    farPlane=0.229372, width=0.016686, height=0.00836624, cameraPosition=(
    0.00833013, -0.00270423, 0.226244), cameraTarget=(0.00833013, -0.00270423, 
    0))
s2.ObliqueDimension(vertex1=v[0], vertex2=v[3], textPoint=(0.00947112217545509, 
    -0.00583955924957991), value=0.0009)
d[4].setValues(value=0.002, )
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.212026, 
    farPlane=0.240462, width=0.0858398, height=0.0430396, cameraPosition=(
    0.0228474, -0.00957759, 0.226244), cameraTarget=(0.0228474, -0.00957759, 
    0))
s2.unsetPrimaryObject()
p = mdb.models['Zeides_config'].parts['Die_2a_8_d0_36mm']
p.features['2D Analytic rigid shell-1'].setValues(sketch=s2)
del mdb.models['Zeides_config'].sketches['__edit__']
p = mdb.models['Zeides_config'].parts['Die_2a_8_d0_36mm']
p.regenerate()
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.00704854, 
    farPlane=0.0136814, width=0.0180719, height=0.0085723, 
    viewOffsetX=0.00516004, viewOffsetY=-0.00180667)
p1 = mdb.models['Zeides_config'].parts['Die_2a_8_d0_36mm']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
p1 = mdb.models['Zeides_config'].parts['Die_2a_8_d0_36mm']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
a = mdb.models['Zeides_config'].rootAssembly
a.regenerate()
session.viewports['Viewport: 1'].setValues(displayedObject=a)
p1 = mdb.models['Zeides_config'].parts['Die_2a_8_d0_36mm']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
p = mdb.models['Zeides_config'].parts['Die_2a_8_d0_36mm']
s = p.features['2D Analytic rigid shell-1'].sketch
mdb.models['Zeides_config'].ConstrainedSketch(name='__edit__', objectToCopy=s)
s1 = mdb.models['Zeides_config'].sketches['__edit__']
g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
s1.setPrimaryObject(option=SUPERIMPOSE)
p.projectReferencesOntoSketch(sketch=s1, 
    upToFeature=p.features['2D Analytic rigid shell-1'], filter=COPLANAR_EDGES)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.23167, 
    farPlane=0.245257, width=0.0362409, height=0.018171, cameraPosition=(
    0.00769463, 0.00623986, 0.238464), cameraTarget=(0.00769463, 0.00623986, 
    0))
d[1].setValues(value=0.017, )
s1.unsetPrimaryObject()
p = mdb.models['Zeides_config'].parts['Die_2a_8_d0_36mm']
p.features['2D Analytic rigid shell-1'].setValues(sketch=s1)
del mdb.models['Zeides_config'].sketches['__edit__']
p = mdb.models['Zeides_config'].parts['Die_2a_8_d0_36mm']
p.regenerate()
a = mdb.models['Zeides_config'].rootAssembly
a.regenerate()
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.263979, 
    farPlane=0.27641, width=0.0332063, height=0.0157512, 
    viewOffsetX=0.000260089, viewOffsetY=-0.0519807)
p1 = mdb.models['Zeides_config'].parts['Die_2a_8_d0_36mm']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
p = mdb.models['Zeides_config'].parts['Die_2a_8_d0_36mm']
s = p.features['2D Analytic rigid shell-1'].sketch
mdb.models['Zeides_config'].ConstrainedSketch(name='__edit__', objectToCopy=s)
s2 = mdb.models['Zeides_config'].sketches['__edit__']
g, v, d, c = s2.geometry, s2.vertices, s2.dimensions, s2.constraints
s2.setPrimaryObject(option=SUPERIMPOSE)
p.projectReferencesOntoSketch(sketch=s2, 
    upToFeature=p.features['2D Analytic rigid shell-1'], filter=COPLANAR_EDGES)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.231288, 
    farPlane=0.24668, width=0.0410555, height=0.020585, cameraPosition=(
    0.00847914, 0.00977163, 0.238984), cameraTarget=(0.00847914, 0.00977163, 
    0))
d[1].setValues(value=0.0175, )
s2.unsetPrimaryObject()
p = mdb.models['Zeides_config'].parts['Die_2a_8_d0_36mm']
p.features['2D Analytic rigid shell-1'].setValues(sketch=s2)
del mdb.models['Zeides_config'].sketches['__edit__']
p = mdb.models['Zeides_config'].parts['Die_2a_8_d0_36mm']
p.regenerate()
a = mdb.models['Zeides_config'].rootAssembly
a.regenerate()
session.viewports['Viewport: 1'].setValues(displayedObject=a)
a = mdb.models['Zeides_config'].rootAssembly
a.translate(instanceList=('Die_2a_8_d0_36mm-1', ), vector=(0.0, -0.006, 0.0))
#: The instance Die_2a_8_d0_36mm-1 was translated by 0., -6.E-03, 0. with respect to the assembly coordinate system
session.viewports['Viewport: 1'].assemblyDisplay.setValues(interactions=ON, 
    constraints=ON, connectors=ON, engineeringFeatures=ON)
a = mdb.models['Zeides_config'].rootAssembly
region1=a.instances['Die_2a_8_d0_36mm-1'].surfaces['For_contact']
a = mdb.models['Zeides_config'].rootAssembly
region2=a.instances['Wire-1'].surfaces['For_contact']
mdb.models['Zeides_config'].SurfaceToSurfaceContactStd(name='Int-1', 
    createStepName='Step-1', main=region1, secondary=region2, sliding=FINITE, 
    thickness=ON, interactionProperty='IntProp-1', adjustMethod=NONE, 
    initialClearance=OMIT, datumAxis=None, clearanceRegion=None)
#: The interaction "Int-1" has been created.
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
    predefinedFields=ON, interactions=OFF, constraints=OFF, 
    engineeringFeatures=OFF)
a = mdb.models['Zeides_config'].rootAssembly
region = a.instances['Wire-1'].sets['Fix_wire']
mdb.models['Zeides_config'].YsymmBC(name='BC-1', createStepName='Step-1', 
    region=region, localCsys=None)
a = mdb.models['Zeides_config'].rootAssembly
region = a.instances['Die_2a_8_d0_36mm-1'].sets['Vel_RP']
mdb.models['Zeides_config'].VelocityBC(name='BC-2', createStepName='Step-1', 
    region=region, v1=0.0, v2=0.016, vr3=0.0, amplitude=UNSET, localCsys=None, 
    distributionType=UNIFORM, fieldName='')
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.236723, 
    farPlane=0.303666, width=0.179962, height=0.085364, viewOffsetX=0.02953, 
    viewOffsetY=-0.0370469)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=OFF, bcs=OFF, 
    predefinedFields=OFF, connectors=OFF)
    
    
    
#mdb.Job(name='Test_job_wd_ml_0', model='Zeides_config', description='', 
#    type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None, 
#    memory=90, memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
#    explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
#    modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
#    scratch='', resultsFormat=ODB, numThreadsPerMpiProcess=1, 
#    multiprocessingMode=DEFAULT, numCpus=2, numDomains=2, numGPUs=0)
"""
mdb.jobs['Test_job_wd_ml_0'].submit(consistencyChecking=OFF)
#: The job input file "Test_job_wd_ml_0.inp" has been submitted for analysis.
#: Job Test_job_wd_ml_0: Analysis Input File Processor completed successfully.
o3 = session.openOdb(name='E:/Calculations_Abaqus_2022/Test_job_wd_ml_0.odb')
#: Model: E:/Calculations_Abaqus_2022/Test_job_wd_ml_0.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     2
#: Number of Meshes:             2
#: Number of Element Sets:       4
#: Number of Node Sets:          6
#: Number of Steps:              1
session.viewports['Viewport: 1'].setValues(displayedObject=o3)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.28206, 
    farPlane=0.419156, width=0.0365946, height=0.0173584, 
    viewOffsetX=0.00899991, viewOffsetY=-0.0375823)
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=0 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=122 )
session.viewports['Viewport: 1'].view.setValues(nearPlane=2.09791, 
    farPlane=2.22711, width=0.0462731, height=0.0219494, cameraPosition=(
    0.00939651, 0.0487371, 2.16251), viewOffsetX=0.0120517, 
    viewOffsetY=-0.0411825)
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=0 )
mdb.jobs['Test_job_wd_ml_0'].kill()
#: Error in job Test_job_wd_ml_0: Process terminated by external request (SIGTERM or SIGINT received).
#: Job Test_job_wd_ml_0: Abaqus/Standard was terminated prior to analysis completion.
#: Error in job Test_job_wd_ml_0: Abaqus/Standard Analysis exited with an error - Please see the  message file for possible error messages if the file exists.
mdb.saveAs(
    pathName='E:/Calculations_Abaqus_2022/Grebenkin/test_wd_cae/Conical_die_AISI_1012_Zeides_clear_one_1.cae')
#: The model database has been saved to "E:\Calculations_Abaqus_2022\Grebenkin\test_wd_cae\Conical_die_AISI_1012_Zeides_clear_one_1.cae".
"""