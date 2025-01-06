import math

from abaqus import mdb
from abaqusConstants import (
    ANALYTIC_RIGID_SURFACE,
    AXISYM,
    AXISYMMETRIC,
    DEFAULT,
    FINITE,
    FRACTION,
    HARD,
    ISOTROPIC,
    KINEMATIC,
    OFF,
    OMIT,
    ON,
    PENALTY,
    STANDALONE,
)

NUM_CPUS = 1

die_vel_rp_name = "Die_2a_24_d0_36mm-1.Vel_RP"
die_inst_name = "Die_2a_24_d0_36mm-1"

d_0 = 0.036  # float(36/100/10)
r_0 = 0.018  # d_0/2
# reductions = [1.5, 2.5, 5., 10., 15., 20., 25., 30.]
# cal_parts = [0., 0.1, 0.3, 0.5, 0.75, 1.]
# friction =[0.05]#, 0.05, 0.025  ]
# velocities = [20]#[1, 2, 5, 10, 20, 30]  #m/min
# half_angles = [4, 8, 12, 16, 20]


reds_ = [1.5, 2.5, 5.0, 10.0, 20.0]
reductions = [17.0, 19.0]

cals_ = [0.0, 0.1, 0.3, 0.5, 0.75, 1.0]
cal_parts = [0.4, 0.9, 1.25]
half_angles = [14, 18, 22]  # [4, 8, 12, 16, 20]
velocities = [1, 2, 5, 10]  # [1, 2, 5, 10, 20, 30]  #m/min
velocities = [20]
friction = [0.075, 0.08]

cp_def = 0.0001

# reductions = [5]
labels = []

for red in reductions:
    r_1 = r_0 - r_0 * red / 100
    print(red, r_1, (1 - red / 100))
    red_100 = int(red * 100)
    for cp in cal_parts:
        cp_len = 2 * r_1 * cp
        cp_100 = int(100 * cp)
        for half_angle in half_angles:
            for vel in velocities:
                full_angle = half_angle * 2
                current_die_name = "Die_2a_" + str(full_angle) + "_d0_36"
                current_die_instance_name = "Die_2a_" + str(full_angle) + "_d0_36-1"
                label = (
                    "zeides_2a_"
                    + str(full_angle)
                    + "_red_"
                    + str(red_100)
                    + "_cal_"
                    + str(cp_100)
                    + "_vel_"
                    + str(vel)
                )
                label = str(label)
                mdb.Model(name=str(label), objectToCopy=mdb.models["Zeides_config"])

                # Setting geometry--------------------------------------------------------------------------------------------------
                s = mdb.models[label].ConstrainedSketch(name="__profile__", sheetSize=200.0)
                g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
                s.sketchOptions.setValues(viewStyle=AXISYM)

                s.setPrimaryObject(option=STANDALONE)
                s.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, 100.0))

                offset = (-0.0583483300576789 - 0.0189852677285671) / 30 * red
                s.Line(point1=(r_1, offset + 0.0), point2=(r_1, offset - cp * d_0))
                s.Line(
                    point1=(r_1, offset - cp * d_0),
                    point2=(
                        r_1 - r_1 * math.sin(math.radians(180 + half_angle)),
                        offset - cp * d_0 - d_0 * math.sin(math.radians(180 - half_angle)),
                    ),
                )
                # s.Line(point1=(r_1, offset+0.0), point2=(1.25*(r_1+d_0*math.sin(math.radians(half_angle))), offset+1.25*(d_0*math.cos(math.radians(half_angle)))))
                s.Line(
                    point1=(r_1, offset + 0.0),
                    point2=(
                        r_1 + 2.5 * d_0 * math.sin(math.radians(half_angle)),
                        offset + 2.5 * d_0 * math.cos(math.radians(half_angle)),
                    ),
                )

                p = mdb.models[label].Part(
                    name=current_die_name, dimensionality=AXISYMMETRIC, type=ANALYTIC_RIGID_SURFACE
                )
                p = mdb.models[label].parts[current_die_name]
                p.AnalyticRigidSurf2DPlanar(sketch=s)
                s.unsetPrimaryObject()

                v1, e, d1, n = p.vertices, p.edges, p.datums, p.nodes
                # --------------------------------------------------------------------------------------------------

                # RP & Normals --------------------------------------------------------------------------------------------------

                if cp_len == 0:
                    mdb.models[label].parts[current_die_name].ReferencePoint(
                        point=mdb.models[label].parts[current_die_name].vertices[0]
                    )

                    rpk = mdb.models[label].parts[current_die_name].referencePoints.keys()[0]

                    mdb.models[label].parts[current_die_name].Set(
                        name="Vel_RP",
                        referencePoints=(
                            mdb.models[label].parts[current_die_name].referencePoints[rpk],
                        ),
                    )
                    mdb.models[label].parts[current_die_name].Surface(
                        name="For_contact",
                        side1Edges=mdb.models[label]
                        .parts[current_die_name]
                        .edges.findAt(
                            p.vertices[-3].pointOn,
                        ),
                    )

                else:
                    # change normals

                    mdb.models[label].parts[current_die_name].Surface(
                        name="For_contact",
                        side2Edges=mdb.models[label]
                        .parts[current_die_name]
                        .edges.findAt(
                            p.vertices[-3].pointOn,
                        ),
                    )

                    mdb.models[label].parts[current_die_name].ReferencePoint(
                        point=mdb.models[label].parts[current_die_name].vertices[3]
                    )

                    rpk = mdb.models[label].parts[current_die_name].referencePoints.keys()[0]
                    # change RP
                    mdb.models[label].parts[current_die_name].Set(
                        name="Vel_RP",
                        referencePoints=(
                            mdb.models[label].parts[current_die_name].referencePoints[rpk],
                        ),
                    )

                # interactionProperties    tangentialBehavior
                mdb.models[label].interactionProperties["IntProp-1"].tangentialBehavior.setValues(
                    dependencies=0,
                    directionality=ISOTROPIC,
                    elasticSlipStiffness=None,
                    formulation=PENALTY,
                    fraction=0.005,
                    maximumElasticSlip=FRACTION,
                    pressureDependency=OFF,
                    shearStressLimit=None,
                    slipRateDependency=OFF,
                    table=((0.05,),),
                    temperatureDependency=OFF,
                )

                # interactionProperties NormalBehavior
                mdb.models[label].interactionProperties["IntProp-1"].NormalBehavior(
                    allowSeparation=ON,
                    constraintEnforcementMethod=DEFAULT,
                    pressureOverclosure=HARD,
                )

                a = mdb.models[label].rootAssembly
                p = mdb.models[label].parts[current_die_name]
                a.Instance(name=current_die_instance_name, part=p, dependent=ON)
                a = mdb.models[label].rootAssembly
                if "Die_2a_24_d0_36mm-1" in a.features.keys():
                    del a.features["Die_2a_24_d0_36mm-1"]

                region1 = a.instances[current_die_instance_name].surfaces["For_contact"]
                mdb.models[label].interactions["Int-1"].setValues(
                    main=region1,
                    mechanicalConstraint=KINEMATIC,
                    sliding=FINITE,
                    interactionProperty="IntProp-1",
                    initialClearance=OMIT,
                    datumAxis=None,
                )

                # Instance
                mdb.models[label].interactions["Int-2"].setValues(
                    clearanceRegion=None,
                    datumAxis=None,
                    initialClearance=OMIT,
                    interactionProperty="IntProp-1",
                    main=mdb.models[label]
                    .rootAssembly.instances[current_die_instance_name]
                    .surfaces["For_contact"],
                    mechanicalConstraint=KINEMATIC,
                    sliding=FINITE,
                )

                # VelocityBC
                a = mdb.models[label].rootAssembly
                region = a.instances[current_die_instance_name].sets["Vel_RP"]
                mdb.models[label].boundaryConditions["BC-2"].setValues(region=region)
                velo_v2 = vel * 0.016666666667  # from m/min to m/s
                mdb.models[label].boundaryConditions["BC-2"].setValues(v2=velo_v2)

                # Step Time Edit
                # ---------------------------------------------------------------------------------------------------------------
                len_base = 0.11
                fora = 0.004
                # coef_die = 1.5 + cp*(d_0/len_base)
                elong_coef = (r_0 / r_1) ** 2
                steptime = round(
                    elong_coef * (abs(offset) + len_base + cp * d_0 + fora) / velo_v2, 5
                )
                mdb.models[label].steps["Step-1"].setValues(timePeriod=steptime)
                # ---------------------------------------------------------------------------------------------------------------
                labels.append(label)

with open("labels.txt", "w") as f:
    f.write(str(labels))
